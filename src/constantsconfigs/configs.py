from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class _BaseValidatedConfig(BaseModel):
    """
    Validated config with extra='forbid'
    """

    model_config = ConfigDict(extra='forbid')


class SerializableOBj(_BaseValidatedConfig):
    """
    SerializableOBj cfg for import
    """

    target_class: str
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class DataConfig(_BaseValidatedConfig):
    """
    Data config
    """

    batch_size: int = 64
    pin_memory: bool = True
    shuffle: bool = True
    dataset_name_json_train: Path = Path('data/gazeta_train.jsonl')
    dataset_name_json_val: Path = Path('data/gazeta_val.jsonl')
    num_samples_train: int = 52400
    num_samples_val: int = 5265
    task_name: str = 'text_sums'
    pretrained_tokenizer: str = 'DeepPavlov/rubert-base-cased'
