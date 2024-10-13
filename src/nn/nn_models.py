from typing import Dict, Union

import torch

from base.base_nn import ModelSummarizationBase
from torchmetrics import MeanMetric
from transformers import EncoderDecoderModel, GPT2Tokenizer, BertTokenizer

from constantsconfigs.configs import ModelConfig, Metrics
from nn.decoder import Seq2SeqDecoder
from nn.encoder import Seq2SeqEncoder
from metrics.get_metrics import get_metrics


class ModelSummarizationEncDec(ModelSummarizationBase):  # noqa: WPS214
    """
    Gpt2ModelSummarization model
    """

    def __init__(
        self,
        cfg_model: ModelConfig,
        cfg_metrics: Metrics,
        tokenizer_encoder: BertTokenizer,
        tokenizer_decoder: GPT2Tokenizer,
        **kwargs,
    ):
        """
        Init
        :param cfg_model: model config
        :param cfg_metrics: metrics config
        """
        super().__init__()

        self.cfg = cfg_model
        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()

        self._valid_metrics = get_metrics(
            rouge_types=cfg_metrics.rouge_types,
            bert=cfg_metrics.bert,
            lang=cfg_metrics.lang,
            verbose=cfg_metrics.verbose,
            dist_sync_on_step=True,
        )

        self.tokenizer_decoder = tokenizer_decoder
        self.tokenizer_encoder = tokenizer_encoder
        self.model_encoder = Seq2SeqEncoder(
            self.cfg.encoder_model,
            self.tokenizer_encoder,
        )
        self.model_decoder = Seq2SeqDecoder(
            self.cfg.decoder_model,
            self.tokenizer_decoder,
        )

        self.model = EncoderDecoderModel(
            encoder=self.model_encoder,
            decoder=self.model_decoder,
        )

        if cfg_model.set_special_tokens_to_model:
            self.set_special_tokens_model()

        self.save_hyperparameters()

    def set_special_tokens_model(self):
        self.model.config.decoder_start_token_id = self.tokenizer_decoder.bos_token_id
        self.model.config.pad_token_id = self.tokenizer_decoder.pad_token_id


class ModelSummarizationGPT(ModelSummarizationBase):  # noqa: WPS214
    """
    Gpt2ModelSummarization model
    """

    def __init__(
        self,
        cfg_model: ModelConfig,
        cfg_metrics: Metrics,
        tokenizer_decoder: GPT2Tokenizer,
        **kwargs,
    ):
        """
        Init
        :param cfg_model: model config
        :param cfg_metrics: metrics config
        """
        super().__init__()

        self.cfg = cfg_model
        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()

        self._valid_metrics = get_metrics(
            rouge_types=cfg_metrics.rouge_types,
            bert=cfg_metrics.bert,
            lang=cfg_metrics.lang,
            verbose=cfg_metrics.verbose,
            dist_sync_on_step=True,
        )

        self.tokenizer = tokenizer_decoder
        self.model = Seq2SeqDecoder(self.cfg.decoder_model, self.tokenizer)

        if cfg_model.set_special_tokens_to_model:
            self.set_special_tokens_model()

        self.save_hyperparameters()

    def validation_step(  # noqa: WPS210
        self,
        batch: Dict[str, Union[torch.Tensor, bool]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        One val step

        :param batch: batch dict(see src.dataprepare.dataset)
        :param batch_idx: batch idx
        :return: logits
        """

        with torch.no_grad():
            logits = self(
                **batch,
            )
        self._valid_loss.update(logits.loss.item())

        generated_ids = torch.argmax(logits.logits, dim=-1)

        for i, index in enumerate(batch["start_index_summary"]):
            generated_ids[i, :index] = self.tokenizer.pad_token_id

        predict_decoded_outputs = self.tokenizer.batch_decode(
            torch.argmax(logits.logits, dim=-1),
            skip_special_tokens=True,
        )
        real_sum = self.tokenizer.batch_decode(
            batch["labels_valid_metric"],
            skip_special_tokens=True,
        )
        self._valid_metrics.update(predict_decoded_outputs, real_sum)

        return logits.loss

    def set_special_tokens_model(self):
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id