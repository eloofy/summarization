from typing import Any, Dict, Union
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
from utils.serialization_module import load_object
from utils.schedulers import get_warmup_scheduler


class ModelSummarizationBase(LightningModule):  # noqa: WPS214
    """
    ModelSummarization base
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.cfg = None
        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()
        self._valid_valid_metrics = None
        self.model = None
        self.tokenizer_encoder = None
        self.tokenizer_decoder = None

    def forward(self, **kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        :param kwargs: batch dict (see src.dataprepare.dataset)
        :return: results
        """
        return self.model(**kwargs)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        One training step
        :param batch: batch dict(see src.dataprepare.dataset)
        :return: loss with logits
        """
        logits = self(
            **batch,
        )
        self._train_loss.update(logits.loss.item())

        self.log(
            "lr",
            self.optimizers().param_groups[0]["lr"],
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "step_loss",
            logits.loss.item(),
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {"loss": logits.loss, "logits": logits}

    def on_train_epoch_end(self) -> None:
        """
        Train epoch end log
        :return: none
        """
        self.log(
            "mean_train_loss",
            self._train_loss.compute(),
            on_step=False,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

        self._train_loss.reset()

    def validation_step(
        self,
        batch: Dict[str, Union[torch.Tensor, bool]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        One val step

        :param batch: batch dict(see src.dataprepare.dataset)
        :param batch_idx: batch idx
        :return: logits
        """

        with torch.no_grad():
            logits = self(**batch)
        self._valid_loss.update(logits.loss.item())

        generated_ids = torch.argmax(logits.logits, dim=-1)
        predict_decoded_outputs = self.tokenizer_decoder.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        real_sum = self.tokenizer_encoder.batch_decode(
            batch["labels"],
            skip_special_tokens=True,
        )
        self._valid_metrics.update(predict_decoded_outputs, real_sum)

        return logits.loss

    def on_validation_epoch_end(self) -> None:
        """
        Val epoch end log
        :return: None
        """
        self.log(
            "mean_valid_loss",
            self._valid_loss.compute(),
            on_step=False,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        self._valid_loss.reset()

        computed_metrics = self._valid_metrics.compute()
        self.log_dict(
            {
                name: value_metric.to(torch.cuda.current_device())
                for name, value_metric in computed_metrics.items()
            },
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        self._valid_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        configure optimizer and schedule

        :return: optimizer and scheduler
        """
        optimizer = load_object(
            self.cfg.optimizer.target_class,
        )(self.model.parameters(), **self.cfg.optimizer.kwargs)

        scheduler = load_object(
            self.cfg.scheduler.target_class,
        )(optimizer, **self.cfg.scheduler.kwargs)
        if self.cfg.num_warmup_steps:
            scheduler = get_warmup_scheduler(
                optimizer,
                self.cfg.num_warmup_steps,
                scheduler,
                self.cfg.optimizer.kwargs["lr"],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.cfg.scheduler_interval,
            },
        }

    def set_special_tokens_model(self):
        """Set special tokens to model config"""