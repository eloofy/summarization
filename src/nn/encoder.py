from typing import Tuple, Union

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from src.constantsconfigs.configs import EncoderConfig


class Seq2SeqEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.model_name = cfg.name_encoder_pretrained_model
        self.config = BertConfig(self.model_name)
        self.model = BertModel.from_pretrained(
            self.model_name,
            self.config,
        )
        self.output_embeddings = None

    def __call__(
        self, **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        output_embeddings = self.forward(
            input_ids=kwargs["input_ids"],
            attention_mask=kwargs["attention_mask"],
        )
        self.output_embeddings = output_embeddings.last_hidden_state.unsqueeze(0)

        return output_embeddings

    def freeze_parameters(self) -> None:
        """
        freeze the parameters of the model
        :return: None
        """
        for param_weight in self.model.parameters():
            param_weight.requires_grad = False

    def unfreeze_parameters(self) -> None:
        """
        Unfreeze the parameters of the model
        :return: None
        """
        for param_weight in self.model.parameters():
            param_weight.requires_grad = True

    def forward(
        self, input_ids, attention_mask, **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        """

        :param input_ids: input ids for encoder (full text)
        :param attention_mask: input mask for encoder
        :param kwargs: additional keyword arguments
        :return: results output
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def get_output_embeddings(self) -> torch.Tensor:  # noqa: WPS615 bug torch
        """
        Get last output model
        :return: output embeddings last layer
        """
        return self.output_embeddings
