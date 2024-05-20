from typing import Tuple, Union

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from src.constantsconfigs.configs import DecoderConfig


class Seq2SeqDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.model_name = cfg.name_base_decoder
        self.config = GPT2Config(
            add_cross_attention=cfg.add_cross_attentions,
            vocab_size=cfg.vocab_size,
            bos_token_id=cfg.bos_token_id,
            eos_token_id=cfg.eos_token_id,
        )
        self.model = GPT2LMHeadModel(self.config)

        if cfg.freeze_layers["freeze_layers"]:
            self.freeze_parameters()

    def __call__(self, **kwargs) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return self.forward(**kwargs)

    def freeze_parameters(self) -> None:
        """
        Freeze the parameters of the model
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
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        Forward masked pass through the decoder model

        :param input_ids: input ids ro decoder (sum text)
        :param attention_mask: input mask for decoder
        :param encoder_hidden_states: encoder hidden states from pretrained model
        :param encoder_attention_mask: encoder attention mask from pretrained model
        :param kwargs: additional keyword arguments
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )
