from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.constantsconfigs.configs import DataConfig
from src.dataprepare.dataset import TextSummarizationDataset
from src.dataprepare.load_data_jsonl2df import load_dataset


class TextSummarizationDatamodule(LightningDataModule):
    def __init__(
        self,
        cfg: DataConfig,
    ) -> None:
        """
        Constructor for TextClassificationDataset class

        :param cfg: data configuration
        """
        super().__init__()
        self.cfg = cfg

        self.tokenizer_encoder = AutoTokenizer.from_pretrained(cfg.pretrained_tokenizer.name_pretrained_tokenizer)
        self.tokenizer_decoder = self.load_tokenizer_decoder()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[TextSummarizationDataset] = None
        self.data_val: Optional[TextSummarizationDataset] = None

    def setup(self, stage: str) -> None:
        """
        Setup data processing

        :param stage: stage of data loading
        :return: None
        """
        if stage == "fit":
            data_train_df = load_dataset(self.cfg.dataset_name_json_train)
            data_val_df = load_dataset(self.cfg.dataset_name_json_val)

            self.data_train = TextSummarizationDataset(
                data_train_df,
                tokenizer_encoder=self.tokenizer_encoder,
                tokenizer_decoder=self.tokenizer_decoder,
                cfg=self.cfg.pretrained_tokenizer,
            )
            self.data_val = TextSummarizationDataset(
                data_val_df,
                tokenizer_encoder=self.tokenizer_encoder,
                tokenizer_decoder=self.tokenizer_decoder,
                cfg=self.cfg.pretrained_tokenizer,
            )

        elif stage == "test":
            return

    def train_dataloader(self):
        """
        Load training dataloader
        :return: training dataloader
        """
        return DataLoader(
            self.data_train,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            pin_memory=self.cfg.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        Load validation dataloader
        :return: validation dataloader
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.batch_size,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """
        Load test dataloader
        :return: test dataloader
        """
        return DataLoader

    def load_tokenizer_decoder(self):
        """
        Load tokenizer decoder with change pad to eos
        :return:
        """
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.pretrained_tokenizer.name_pretrained_tokenizer)
        tokenizer.add_special_tokens(self.cfg.pretrained_tokenizer.special_tokens_add)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer
