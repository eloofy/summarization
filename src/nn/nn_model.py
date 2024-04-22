from typing import Any, Dict

import torch.nn as nn
import torch
import torch.nn.functional as func
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
from transformers import BertConfig, AutoModel

from src.constantsconfigs.configs import ModelConfig
from src.nn.schedulers import get_cosine_schedule_with_warmup
from src.nn.serialization_module import load_object


class BERTModelSummarization(LightningModule):  # noqa: WPS214
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

        self.config_pretrained_model = BertConfig(cfg.pretrained_model)
        self.model = AutoModel.from_pretrained(cfg.pretrained_model)

        self.classifier = nn.Linear(
            self.config_pretrained_model.hidden_size,
            cfg.vocab_size
        )

        self.save_hyperparameters()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        :param batch: barch dict
        :return: results
        """
        outputs = self.model(batch['input_ids'], attention_mask=batch['attention_mask'])
        return self.classifier(outputs.last_hidden_state)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        One training step
        :param batch: texts and labels train
        :return: loss with logits
        """
        logits = self(batch)

        loss = func.cross_entropy(logits, batch['summarization'])
        self._train_loss.update(loss)

        self.log('step_loss', loss, on_step=True, prog_bar=True, logger=True)
        return {'loss': loss, 'logits': logits}

    def on_train_epoch_end(self) -> None:
        """
        Train epoch end log
        :return: none
        """
        self.log(
            'mean_train_loss',
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

        :param batch: texts and labels val
        :param batch_idx: batch idx
        :return: logits
        """
        logits = self(batch)
        loss = func.cross_entropy(logits, batch['summarization'])
        self._valid_loss.update(loss)

        # predictions = torch.argmax(logits, dim=1)
        # self._valid_metrics.update(predictions, batch['label'])

        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Val epoch end log
        :return: none
        """
        self.log(
            'mean_valid_loss',
            self._valid_loss.compute(),
            on_step=False,
            prog_bar=True,
            on_epoch=True,
        )
        self._valid_loss.reset()

        # self.log_dict(self._valid_metrics.compute(), prog_bar=True, on_epoch=True)
        # self._valid_metrics.reset()

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
            num_warmup_steps=2000,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=1.4,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
