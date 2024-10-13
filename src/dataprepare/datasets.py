from typing import Tuple, Dict
import torch

import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer, GPT2Tokenizer

from constantsconfigs.configs import EncoderTokenizer, DecoderTokenizer


class TextSummarizationDatasetEncoderDecoder(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_encoder: BertTokenizer,
        tokenizer_decoder: GPT2Tokenizer,
        cfg_encoder: EncoderTokenizer,
        cfg_decoder: DecoderTokenizer,
        **kwargs,
    ):
        """
        Text Classification Dataset constructor

        :param df: data
        :param tokenizer_encoder: encoder for texts
        :param tokenizer_decoder: decoder for texts
        :param cfg_encoder: tokenizer encoder config
        :param cfg_decoder: tokenizer decoder config
        """
        self.texts = df["text"].values
        self.summarizations = df["summary"].values
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder
        self.cfg_encoder = cfg_encoder
        self.cfg_decoder = cfg_decoder

    def __getitem__(self, idx: int) -> Dict:
        """
        Get tokenized text and labels for train

        :param idx: index of data
        :return: dict
        """
        text, sum_text = self.get_row_data(idx)
        inputs_encoder = self.tokenizer_encoder.encode_plus(
            text,
            max_length=self.cfg_encoder.encoder_max_length,
            padding=self.cfg_encoder.encoder_padding,
            return_tensors="pt",
            truncation=self.cfg_encoder.encoder_truncate,
        )
        inputs_decoder = self.tokenizer_decoder.encode_plus(
            sum_text,
            max_length=self.cfg_decoder.decoder_max_length,
            padding=self.cfg_decoder.decoder_padding,
            return_tensors="pt",
            truncation=self.cfg_decoder.decoder_truncate,
            add_special_tokens=False,
        )

        return {
            "input_ids": inputs_encoder["input_ids"].squeeze(0),
            "attention_mask": inputs_encoder["attention_mask"].squeeze(0),
            "labels": inputs_decoder["input_ids"].squeeze(0),
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


class TextSummarizationDatasetGPT(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_decoder: GPT2Tokenizer,
        cfg_decoder: DecoderTokenizer,
        **kwargs,
    ):
        """
        Text Classification Dataset constructor

        :param df: data
        :param tokenizer: tokenizer for texts
        :param cfg_tokenizer: tokenizer transformer config
        """
        self.texts = df["text"].values
        self.summarizations = df["summary"].values
        self.tokenizer = tokenizer_decoder
        self.cfg_tokenizer = cfg_decoder

    def __getitem__(self, idx: int) -> Dict:  # noqa: WPS210
        """
        Get tokenized text and labels for train

        :param idx: index of data
        :return: dict
        """
        text, sum_text = self.get_row_data(idx)
        inputs = self.tokenizer.encode_plus(
            f"{text} {self.tokenizer.eos_token} <summ> {sum_text}",
            max_length=self.cfg_tokenizer.decoder_max_length,
            padding=self.cfg_tokenizer.decoder_padding,
            return_tensors="pt",
            truncation=self.cfg_tokenizer.decoder_truncate,
            add_special_tokens=True,
        )

        token_sum_pos = torch.where(
            inputs.input_ids[0] == self.tokenizer.added_tokens_encoder["<summ>"],
        )[0]

        labels_valid_metric = inputs.input_ids[0].clone()
        labels_valid_metric[: token_sum_pos.item() + 1] = self.tokenizer.pad_token_id

        labels = inputs.input_ids[0].clone()
        labels[: token_sum_pos.item() + 1] = -100

        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": labels,
            "labels_valid_metric": labels_valid_metric,
            "start_index_summary": token_sum_pos,
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