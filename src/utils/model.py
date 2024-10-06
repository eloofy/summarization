from nn.nn_models import ModelSummarizationEncDec, ModelSummarizationGPT
from dataprepare.datasets import (
    TextSummarizationDatasetEncoderDecoder,
    TextSummarizationDatasetGPT,
)
from dataprepare.datamodules import (
    TextSummarizationDatamoduleEncDec,
    TextSummarizationDatamoduleGPT,
)
from typing import Dict, Type


class SummarizationModel:
    """Loader for summarization models and associated components."""

    def __init__(self):
        self._task_map = self._initialize_task_map()

    @property
    def task_map(self) -> Dict[str, Dict[str, Type]]:
        """Get the task map that associates task types with their components.

        :return dict: The task map with components for each task type.
        """
        return self._task_map

    @classmethod
    def _initialize_task_map(cls) -> Dict[str, Dict[str, Type]]:
        """
        Initialize the task map with model, dataset, and validator/dataloader classes.

        :return dict: A dictionary mapping task types to their corresponding classes.
        """
        return {
            "EncoderDecoder": {
                "model": ModelSummarizationEncDec,
                "dataset": TextSummarizationDatasetEncoderDecoder,
                "datamodule": TextSummarizationDatamoduleEncDec,
            },
            "GPT": {
                "model": ModelSummarizationGPT,
                "dataset": TextSummarizationDatasetGPT,
                "datamodule": TextSummarizationDatamoduleGPT,
            },
        }
