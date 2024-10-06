from typing import Optional, Callable
from transformers import AutoTokenizer

from constantsconfigs.configs import DataConfig
from dataprepare.datasets import TextSummarizationDatasetEncoderDecoder

from base.base_datamodule import TextSummarizationDatamodule
from constantsconfigs.constants import DEFAULT_PROJECT_PATH


class TextSummarizationDatamoduleEncDec(TextSummarizationDatamodule):
    def __init__(
        self,
        cfg: DataConfig,
        task_type: str,
        dataset: Callable,
    ) -> None:
        """
        Constructor for TextSummarizationDatamoduleBase with pretrained encoder and base decoder GPT2

        :param cfg: Data configuration
        :param task_type: Type of dataset to use, either 'encoder_decoder' or 'transformer'
        :param dataset: Dataset task
        """
        super().__init__(cfg, task_type, dataset)
        self.cfg = cfg

        self.tokenizer_encoder = self.load_tokenizer_encoder()
        self.tokenizer_decoder = self.load_tokenizer_decoder()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[TextSummarizationDatasetEncoderDecoder] = None
        self.data_val: Optional[TextSummarizationDatasetEncoderDecoder] = None

    def load_tokenizer_decoder(self):
        """
        Load tokenizer decoder with change pad to eos and create special tokens
        :return:
        """
        tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_PROJECT_PATH / self.cfg.decoder_tokenizer.load_path_tokenizer,
        )
        if self.cfg.decoder_tokenizer.special_tokens:
            tokenizer.add_special_tokens(self.cfg.decoder_tokenizer.special_tokens)

        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def load_tokenizer_encoder(self):
        """
        Load tokenizer encoder
        :return:
        """

        return AutoTokenizer.from_pretrained(
            DEFAULT_PROJECT_PATH / self.cfg.encoder_tokenizer.load_path_tokenizer,
        )


class TextSummarizationDatamoduleGPT(TextSummarizationDatamodule):
    def __init__(
        self,
        cfg: DataConfig,
        task_type: str,
        dataset: Callable,
    ) -> None:
        """
        Constructor for TextSummarizationDatamoduleBase with pretrained encoder and base decoder GPT2

        :param cfg: Data configuration
        :param task_type: Type of dataset to use, either 'encoder_decoder' or 'transformer'
        :param dataset: Dataset task
        """
        super().__init__(cfg, task_type, dataset)

        self.tokenizer_decoder = self.load_tokenizer()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[TextSummarizationDatasetEncoderDecoder]
        self.data_val: Optional[TextSummarizationDatasetEncoderDecoder]

    def load_tokenizer(self):
        """
        Load tokenizer decoder with change pad to eos and create special tokens
        :return:
        """
        tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_PROJECT_PATH / self.cfg.decoder_tokenizer.load_path_tokenizer,
        )
        tokenizer.add_bos_token = True
        if self.cfg.decoder_tokenizer.special_tokens:
            tokenizer.add_special_tokens(self.cfg.decoder_tokenizer.special_tokens)

        if self.cfg.decoder_tokenizer.padding_side:
            tokenizer.padding_side = self.cfg.decoder_tokenizer.padding_side

        return tokenizer
