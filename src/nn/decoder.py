from typing import Tuple, Union

import torch

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from src.constantsconfigs.configs import DecoderConfig
from src.constantsconfigs.constants import DEFAULT_PROJECT_PATH


class Seq2SeqDecoder(GPT2LMHeadModel):
    """
    Decoder model
    """

    def __init__(self, cfg: DecoderConfig, tokenizer: GPT2Tokenizer):
        """
        Decoder init

        :param cfg: cfg decoder model
        :param tokenizer: tokenizer decoder
        """
        self.cfg = cfg
        self.tokenizer = tokenizer

        if self.cfg.pretrained:
            config_model = AutoConfig.from_pretrained(
                cfg.name_decoder_pretrained_model,
                resume_download=None,
            )
            config_model.add_cross_attention = cfg.add_cross_attentions
            super().__init__(config_model)
            self.load_weights()
            self.resize_token_embeddings(len(tokenizer))
        else:
            config_model = GPT2Config(
                vocab_size=len(tokenizer),
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **cfg.gpt2_base_model_params,
            )
            config_model.add_cross_attention = cfg.add_cross_attentions
            super().__init__(config_model)

        self.set_special_tokens()
        self.set_padding_side()

        if cfg.freeze_layers:
            self.freeze_parameters()

    def __call__(self, **kwargs) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        """
        Forward call
        :param kwargs: inputs params
        :return: results
        """
        return self.forward(
            input_ids=kwargs.pop("input_ids", None),
            attention_mask=kwargs.pop("attention_mask", None),
            labels=kwargs.pop("labels", None),
            encoder_hidden_states=kwargs.pop("encoder_hidden_states", None),
            encoder_attention_mask=kwargs.pop("encoder_attention_mask", None),
            return_dict=kwargs.pop("return_dict", None),
        )

    def freeze_parameters(self) -> None:
        """
        Freeze the parameters of the model
        :return: None
        """
        if "embeddings" in self.cfg.freeze_layers:
            self.cfg.freeze_layers.extend(["wte", "wpe"])

        for layer_name, params_w in self.named_parameters():
            if any(  # noqa: WPS337
                layer_freeze in layer_name.split(".")
                for layer_freeze in self.cfg.freeze_layers
            ):
                params_w.requires_grad = False

    def unfreeze_parameters(self) -> None:
        """
        Unfreeze the parameters of the model
        :return: None
        """
        for param_weight in self.parameters():
            param_weight.requires_grad = True

    def load_weights(self):
        """
        Load pretrained weights
        """
        weights = torch.load(DEFAULT_PROJECT_PATH / self.cfg.weights_pretrained_path)
        self.load_state_dict(weights, strict=False)

    def set_special_tokens(self):
        """
        Set special tokens for model (EOS, BOS)
        :return:
        """
        self.config.eos_token_id = self.tokenizer.eos_token_id
        self.config.bos_token_id = self.tokenizer.bos_token_id

    def set_padding_side(self):
        """
        Set padding side
        :return:
        """
        self.config.padding_side = self.cfg.padding_side
