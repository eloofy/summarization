from typing import Tuple

import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer

from src.constantsconfigs.configs import TokenizerConfig


class TextSummarizationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_encoder: BertTokenizer,
        tokenizer_decoder: BertTokenizer,
        cfg: TokenizerConfig,
    ):
        """
        Text Classification Dataset constructor

        :param df: data
        :param tokenizer_encoder: encoder for texts
        :param tokenizer_decoder: decoder for texts
        :param cfg: tokenizer config
        """
        self.tokenizer_config = cfg
        self.texts = df["text"].values
        self.summarizations = df["summary"].values
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder

        self.tokenizer_decoder.pad_token = self.tokenizer_decoder.eos_token

    def __getitem__(self, idx: int) -> dict:
        """
        Get tokenized text and labels for train

        :param idx: index of data
        :return: dict
        """
        text, sum_text = self.get_row_data(idx)
        inputs_encoder = self.tokenizer_encoder.encode_plus(
            text,
            max_length=self.tokenizer_config.encoder_max_length,
            padding=self.tokenizer_config.encoder_padding,
            return_tensors="pt",
            truncation=True,
        )
        inputs_decoder = self.tokenizer_decoder.encode_plus(
            f'{self.tokenizer_config.special_tokens_add["bos_token"]} {sum_text}',
            max_length=self.tokenizer_config.decoder_max_length,
            padding=self.tokenizer_config.decoder_padding,
            return_tensors="pt",
            truncation=self.tokenizer_config.decoder_truncate,
            add_special_tokens=self.tokenizer_config.decoder_add_special_tokens,
        )

        return {
            "input_ids": inputs_encoder["input_ids"].squeeze(0),
            "attention_mask": inputs_encoder["attention_mask"].squeeze(0),
            "decoder_input_ids": inputs_decoder["input_ids"],
            "decoder_attention_mask": inputs_decoder["attention_mask"],
            "labels": inputs_decoder["input_ids"],
        }

    def __len__(self) -> int:
        """
        Get length of dataset
        :return: length of dataset
        """
        return len(self.texts)

    def get_row_data(self, idx: int) -> Tuple[str, str]:
        """
        Get row data from df
        :param idx: index of row data
        :return: text and summarizations
        """
        text = self.texts[idx]
        label = self.summarizations[idx]

        return text, label
