import random
from typing import Any, List

import pytorch_lightning as pl
import six
from clearml import Logger
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import BertTokenizer

from src.constantsconfigs.configs import DebugSamplesConfig, TokenizerConfig
from src.dataprepare.load_data_jsonl2df import load_dataset


class DebugSamples(Callback):
    """
    Class to save txt text generate
    """

    def __init__(
        self,
        cfg: DebugSamplesConfig,
        tokenizer: BertTokenizer,
        cfg_tokenizer: TokenizerConfig,
    ):
        """
        Constructor to save txt text generate
        :param cfg: config for debug samples class
        :param tokenizer: tokenizer
        :param cfg_tokenizer: cfg tokenizer from encoder
        """
        super().__init__()

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.tokenizer_cfg = cfg_tokenizer
        self.data_test = load_dataset(
            self.cfg.data_test_path,
        )
        self.len_data = len(self.data_test)

    def on_train_batch_end(  # noqa: WPS210
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """

        :param trainer: trainer loop obj
        :param pl_module: pl_module obj (dont used)
        :param outputs: outputs batch (dont used)
        :param batch: (dont used)
        :param batch_idx: (dont used)
        :return: None
        """
        if trainer.global_step % self.cfg.each_step != 0:
            return

        random_indexes = [random.randint(0, self.len_data) for _ in range(10)]
        data_text = self.data_test.iloc[random_indexes]["text"]

        results_predict = self.get_predict_sums(
            trainer,
            data_text.values,
        )

        save_result_str = "Results: \n"

        for text, summary in zip(data_text.values, results_predict):  # noqa: WPS519
            save_result_str += f"{text}: {summary}\n\n"  # noqa: WPS336

        Logger.current_logger().report_media(
            title=f"Epoch: {trainer.current_epoch} Iteration: {trainer.global_step}",
            series="ignored",
            iteration=trainer.global_step,
            stream=six.StringIO(save_result_str),
            file_extension=".txt",
        )

    def get_predict_sums(
        self,
        trainer: "pl.Trainer",
        data_text: List[str],
    ) -> List[str]:
        """
        Method to generate predict sums

        :param trainer:
        :param data_text:
        :return: predicted results
        """
        input_ids = self.tokenizer.encode_plus(
            data_text,
            max_length=self.tokenizer_cfg.encoder_max_length,
            padding=self.tokenizer_cfg.encoder_padding,
            return_tensors="pt",
            truncation=True,
        ).input_ids

        predict_tokens = trainer.model.generate(
            input_ids,
        )

        return self.tokenizer.batch_decode(
            predict_tokens,
            skip_special_tokens=True,
        )