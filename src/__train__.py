import warnings
import sys
import os
import pytorch_lightning as pl
import torch
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from clearml import Task

from src.constantsconfigs.configs import ExperimentConfig
from src.constantsconfigs.constants import DEFAULT_PROJECT_PATH
from utils.model import SummarizationModel
from pathlib import Path
from src.utils.load_config import load_config

warnings.simplefilter(action="ignore", category=FutureWarning)  # bag
warnings.simplefilter(
    action="ignore",
    category=UserWarning,
)  # warning from transformers
sys.setrecursionlimit(2048)  # noqa: WPS432 for big seq
torch.set_float32_matmul_precision("medium")


def train(cfg: ExperimentConfig, mlflow_cfg_file: Path) -> None:  # noqa: WPS210
    """
    Train loop

    :param cfg: file with exp config
    :param mlflow_cfg_file: path to clearml server cfg file
    """
    pl.seed_everything(0)
    if mlflow_cfg_file:
        os.environ["CLEARML_CONFIG_FILE"] = str(mlflow_cfg_file)
        Task.init(project_name=cfg.project_name, task_name=cfg.experiment_name)

    sums_map = SummarizationModel().task_map[cfg.task]
    datamodule = sums_map["datamodule"](
        cfg=cfg.data_config,
        task_type=cfg.task,
        dataset=sums_map["dataset"],
    )
    model = sums_map["model"](
        cfg_model=cfg.pr_model_config,
        cfg_metrics=cfg.metrics,
        tokenizer_encoder=datamodule.tokenizer_encoder,
        tokenizer_decoder=datamodule.tokenizer_decoder,
    )
    callbacks = [
        ModelCheckpoint(
            filename="BERT-{epoch}--{mean_valid_loss:.5f}",
            save_top_k=1,
            monitor="mean_valid_loss",
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        **dict(cfg.trainer_config),
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_exp_cfg",
        default=DEFAULT_PROJECT_PATH
        / Path("../configs/GPT_PRETRAINED/exp_config.yaml"),
        type=Path,
    )
    parser.add_argument("--mlflow_cfg_file", default=None, type=Path)
    args = parser.parse_args()
    exp_cfg = load_config(args.path_exp_cfg)

    train(cfg=exp_cfg, mlflow_cfg_file=args.mlflow_cfg_file)
