from typing import Any, Dict

import torch
import torch.nn.functional as func
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
from transformers import EncoderDecoderModel

from src.constantsconfigs.configs import ModelConfig
from src.nn.decoder import Seq2SeqDecoder
from src.nn.encoder import Seq2SeqEncoder
from src.nn.schedulers import get_cosine_schedule_with_warmup
from src.nn.serialization_module import load_object


class Gpt2ModelSummarization(LightningModule):  # noqa: WPS214
    """
    Vit model
    """

    def __init__(
        self,
        cfg: ModelConfig,
    ):
        """
        Init
        :param cfg: module config
        """
        super().__init__()

        self.cfg = cfg
        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()

        self.model_encoder = Seq2SeqEncoder(self.cfg.encoder)
        self.model_decoder = Seq2SeqDecoder(self.cfg.decoder)

        self.model = EncoderDecoderModel(
            encoder=self.model_encoder,
            decoder=self.model_decoder,
        )

        self.save_hyperparameters()

    def forward(self, batch: Dict[str, str]) -> torch.Tensor:
        """
        Forward pass
        :param batch: batch dict(see src.dataprepare.dataset)
        :return: results
        """
        return self.model(**batch)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        One training step
        :param batch: batch dict(see src.dataprepare.dataset)
        :return: loss with logits
        """
        logits = self(batch)

        loss = func.cross_entropy(
            logits.logits[0][0],
            batch["labels"].view(-1),
        )
        self._train_loss.update(loss.item())

        self.log("step_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "logits": logits}

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
        )
        self._train_loss.reset()

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        One training step

        :param batch: bach dict(see src.dataprepare.dataset)
        :param batch_idx: batch idx
        :return: logits
        """
        logits = self(batch)
        loss = func.cross_entropy(
            logits.logits[0][0],
            batch["labels"].view(-1),
        )
        self._valid_loss.update(loss.item())

        return loss

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
        )
        self._valid_loss.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        configure optimizer and schedule

        :return: optimizer and scheduler
        """
        optimizer = load_object(
            self.cfg.optimizer.target_class,
        )(self.model.parameters(), **self.cfg.optimizer.kwargs)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=self.cfg.num_cycles,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
