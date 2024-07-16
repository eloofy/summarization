import torch
from constantsconfigs.constants import DEFAULT_PROJECT_PATH
from typing import Tuple, Union
import torch.nn as nn
from transformers import BertConfig, BertModel
from constantsconfigs.configs import EncoderConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)  # isort: skip
from transformers import BertTokenizer


class Seq2SeqEncoder(BertModel):
    """
    Model encoder
    """

    def __init__(self, cfg: EncoderConfig, encoder_tokenizer: BertTokenizer):
        """
        Encoder init.

        :param cfg: config encoder
        :param encoder_tokenizer: encoder tokenizer
        """
        self.cfg = cfg
        self.encoder_tokenizer = encoder_tokenizer

        if self.cfg.pretrained:
            config_model = BertConfig.from_pretrained(
                cfg.name_encoder_pretrained_model,
                resume_download=None,
            )
            super().__init__(config_model, add_pooling_layer=False)
            self.load_weights()
        else:
            config_model = BertConfig(
                vocab_size=encoder_tokenizer.vocab_size,
                **cfg.bert_base_model_params,
            )
            super().__init__(config_model, add_pooling_layer=False)

        self.projection_layer = None
        if not self.check_hid_size_encdec():
            self.create_projection_layer()

        if cfg.freeze_layers:
            self.freeze_parameters()

    def __call__(
        self,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        input_ids = kwargs.pop("input_ids")
        attention_mask = kwargs.pop("attention_mask")

        bert_outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self.projection_layer:
            bert_outputs.last_hidden_state = self.projection_layer(
                bert_outputs.last_hidden_state,
            )

        return bert_outputs

    def freeze_parameters(self) -> None:
        """
        freeze the parameters of the model
        :return: None
        """
        if "all" in self.cfg.freeze_layers:
            for param_weight in self.parameters():
                param_weight.requires_grad = False

            return

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

    def create_projection_layer(self):
        """
        Create projection layer
        :return:
        """
        self.projection_layer = nn.Linear(
            self.config.hidden_size,
            self.cfg.decoder_hid_size,
        )
        self.config.hidden_size = self.cfg.decoder_hid_size

    def check_hid_size_encdec(self):
        """
        Checking hidden state matching encdec
        """
        return self.config.hidden_size == self.cfg.decoder_hid_size
