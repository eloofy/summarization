import logging
from typing import Optional, Union, Callable

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from constantsconfigs.configs import DataConfig

from transformers import GPT2Tokenizer, BertTokenizer

from dataprepare.datasets import (
    TextSummarizationDatasetEncoderDecoder,
    TextSummarizationDatasetGPT,
)
from utils.load_data_jsonl2df import load_dataset
import pandas as pd

logger = logging.getLogger(__name__)


class TextSummarizationDatamodule(LightningDataModule):  # noqa: WPS230
    def __init__(self, cfg: DataConfig, task_type: str, dataset: Callable):
        """
        Datamodule for text summarization tasks using PyTorch Lightning.

        :param cfg: Configuration dictionary containing dataset and tokenizer settings
        :param task_type: Type of dataset to use, either 'encoder_decoder' or 'transformer'
        :param dataset: dataset task
        """
        super().__init__()
        self.cfg = cfg
        self.task_type = task_type
        self.dataset = dataset

        self.tokenizer_model: Optional[Union[GPT2Tokenizer]] = None
        self.tokenizer_encoder: Optional[Union[GPT2Tokenizer, BertTokenizer]] = None
        self.tokenizer_decoder: Optional[Union[GPT2Tokenizer]] = None
        self.data_train: Optional[
            Union[TextSummarizationDatasetEncoderDecoder, TextSummarizationDatasetGPT]
        ] = None
        self.data_val: Optional[
            Union[TextSummarizationDatasetEncoderDecoder, TextSummarizationDatasetGPT]
        ] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup data processing for different stages.

        :param stage: Stage of data loading, either 'fit' or 'test'
        """
        logger.info(f"Setting up data for stage: {stage}")
        if stage == "fit" or stage is None:
            data_train_df = pd.concat(
                [
                    load_dataset(self.cfg.dataset_name_json_train_first),
                    load_dataset(self.cfg.dataset_name_json_train_second),
                ],
            )
            data_val_df = load_dataset(self.cfg.dataset_test_val)

            self.data_train = self.dataset(
                data_train_df,
                tokenizer_encoder=self.tokenizer_encoder,
                tokenizer_decoder=self.tokenizer_decoder,
                tokenizer=self.tokenizer_model,
                cfg_encoder=self.cfg.encoder_tokenizer,
                cfg_decoder=self.cfg.decoder_tokenizer,
            )

            self.data_val = self.dataset(
                data_val_df,
                tokenizer_encoder=self.tokenizer_encoder,
                tokenizer_decoder=self.tokenizer_decoder,
                tokenizer=self.tokenizer_model,
                cfg_encoder=self.cfg.encoder_tokenizer,
                cfg_decoder=self.cfg.decoder_tokenizer,
            )
            logger.info("Data setup completed.")

    def train_dataloader(self) -> DataLoader:
        """
        Load training dataloader.

        :return: Training dataloader
        """
        logger.info("Creating training dataloader.")
        return DataLoader(
            self.data_train,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            pin_memory=self.cfg.pin_memory,
            drop_last=True,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Load validation dataloader.

        :return: Validation dataloader
        """
        logger.info("Creating validation dataloader.")
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.batch_size,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

    def load_tokenizer(self):
        """If not EncoderDecoder model setup one tokenizer"""

    def load_tokenizer_decoder(self):
        """Load tokenizer decoder"""

    def load_tokenizer_encoder(self):
        """Load tokenizer encoder"""