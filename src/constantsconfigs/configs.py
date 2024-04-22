from pathlib import Path
from typing import Any, Dict, Optional, List, Literal

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


class ModelConfig(_BaseValidatedConfig):
    name_model: str = 'BERT'
    pretrained_model: str = 'DeepPavlov/rubert-base-cased'
    optimizer: SerializableOBj = SerializableOBj(
        target_class='torch.optim.AdamW',
        kwargs={'lr': 1e-4, 'weight_decay': 1e-1},
    )
    vocab_size: int = 119547


class TrainerConfig(_BaseValidatedConfig):
    """
    Trainer config
    """

    min_epochs: int = 20
    max_epochs: int = 30
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 1
    gradient_clip_val: Optional[float] = 0.1
    gradient_clip_algorithm: Optional[Literal['norm', 'value']] = 'norm'
    deterministic: bool = False
    fast_dev_run: bool = False
    default_root_dir: Optional[Path] = None
    detect_anomaly: bool = False
    accelerator: str = 'gpu'
    devices: List = [0]


class ExperimentConfig(_BaseValidatedConfig):
    """
    Experiment config
    """

    project_name: str = 'BERTSummarization'
    experiment_name: str = 'exp_main_pretrained_bert'
    trainer_config: TrainerConfig = Field(default=TrainerConfig())
    data_config: DataConfig = Field(default=DataConfig())
    module_config: ModelConfig = Field(default=ModelConfig())

