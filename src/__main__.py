import os
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from src.constantsconfigs.configs import ExperimentConfig
from src.constantsconfigs.constants import DEFAULT_PROJECT_PATH
from src.dataprepare.datamodule import TextSummarizationDatamodule
from src.nn.nn_model_main import Gpt2ModelSummarization

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


def train(cfg: ExperimentConfig) -> None:  # noqa: WPS210
    """
    Train loop

    :param cfg: file with exp config
    :return:
    """
    pl.seed_everything(0)
    datamodule = TextSummarizationDatamodule(cfg=cfg.data_config)
    callbacks = [
        ModelCheckpoint(
            filename="BERT-{epoch}--{mean_valid_loss:.4f}",
            save_top_k=1,
            monitor="mean_valid_loss",
            mode="min",
            every_n_epochs=5,
        ),
    ]

    model = Gpt2ModelSummarization(cfg=cfg.config_full_model)
    trainer = pl.Trainer(callbacks=callbacks, **dict(cfg.trainer_config))
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    cfg_model_path_yaml = os.getenv(
        "TRAIN_CFG_PATH",
        DEFAULT_PROJECT_PATH / "configs" / "train.yaml",
    )
    train(cfg=ExperimentConfig())
