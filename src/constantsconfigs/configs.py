from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _BaseValidatedConfig(BaseModel):
    """
    Validated config with extra='forbid'
    """

    model_config = ConfigDict(extra="forbid")


class SerializableOBj(_BaseValidatedConfig):
    """
    SerializableOBj cfg for import
    """

    target_class: str
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class TokenizerConfig(_BaseValidatedConfig):
    """
    TokenizerConfig cfg for

    :var name_pretrained_tokenizer: name of pretrained tokenizer
    :var encoder_max_length: max length of encoder
    :var encoder_padding: encoder padding
    :var encoder_truncate: encoder truncate
    :var decoder_max_length: max length of decoder
    :var decoder_padding: decoder padding
    :var decoder_truncate: decoder truncate
    :var decoder_add_special_tokens: decoder special tokens add
    :var special_tokens_add: special tokens

    """
    name_pretrained_tokenizer: str = 'DeepPavlov/rubert-base-cased'
    encoder_max_length: int = 512
    encoder_padding: str = 'max_length'
    encoder_truncate: bool = True
    decoder_max_length: int = 128
    decoder_padding: str = 'max_length'
    decoder_truncate: bool = True
    decoder_add_special_tokens: bool = False
    special_tokens_add: Dict = {'bos_token': '[BOS]', 'eos_token': '[EOS]'}


class EncoderConfig(_BaseValidatedConfig):
    """
    Encoder config

    :var name_encoder_pretrained_model: name of pretrained model for encoder
    :var freeze_layers: dict for freeze layers

    """
    name_encoder_pretrained_model: str = "DeepPavlov/rubert-base-cased"
    freeze_layers: Dict[str, Any] = {
        'freeze_layers': False,
        'layers': 5,
    }


class DecoderConfig(_BaseValidatedConfig):
    """
    Encoder config

    :var add_cross_attentions: add cross attention layers
    :var vocab_size: vocab size decoder tokenizer
    :var bos_token_id: bos token id
    :var eos_token_id: eos token id
    :var name_base_decoder: name of model for decoder
    :var freeze_layers: dict for freeze layers
    """
    add_cross_attentions: bool = True
    vocab_size: int = 119549
    bos_token_id: int = 119547
    eos_token_id: int = 119548
    name_base_decoder: str = "gpt2"
    freeze_layers: Dict[str, Any] = {
        'freeze_layers': False,
        'layers': 5,
    }


class DataConfig(_BaseValidatedConfig):
    """
    Data config

    :var batch_size: batch size
    :var pin_memory: pin memory
    :var shuffle: shuffle data
    :var dataset_name_json_train: path to train dataset json
    :var dataset_name_json_val: path to validation dataset json
    :var encoder_model: encoder model object
    :var decoder_model: decoder model object
    :var num_samples_train: number of training samples
    :var num_samples_val: number of validation samples
    :var pretrained_tokenizer: base pretrained tokenizer
    """

    batch_size: int = 64
    pin_memory: bool = True
    shuffle: bool = True
    dataset_name_json_train: Path = Path("data/gazeta_train.jsonl")
    dataset_name_json_val: Path = Path("data/gazeta_val.jsonl")
    encoder_model: EncoderConfig = Field(default=EncoderConfig())
    decoder_model: DecoderConfig = Field(default=DecoderConfig())
    num_samples_train: int = 52400
    num_samples_val: int = 5265
    pretrained_tokenizer: TokenizerConfig = Field(default=TokenizerConfig())


class ModelConfig(_BaseValidatedConfig):
    """
    Model config full

    :var name_model_full: base model name
    :var encoder: encoder model object
    :var decoder: decoder model object
    :var optimizer: optimizer object
    :var num_warmup_steps: number of warmup steps
    :var num_cycles: cycle
    """
    name_model_full: str = "gpt_2_encoder_pretrained"
    encoder: EncoderConfig = Field(default=EncoderConfig())
    decoder: DecoderConfig = Field(default=DecoderConfig())
    optimizer: SerializableOBj = SerializableOBj(
        target_class="torch.optim.AdamW",
        kwargs={"lr": 1e-4, "weight_decay": 1e-1},
    )
    num_warmup_steps: int = 2000
    num_cycles: float = 1.4


class TrainerConfig(_BaseValidatedConfig):
    """
    Trainer config

    :var min_epochs: min trained epochs until use patience
    :var max_epochs: max trained epochs
    :var check_val_every_n_epoch: check val every n epochs
    :var log_every_n_steps: log metrics every n steps
    :var gradient_clip_val: clipping value for gradient clipping
    :var deterministic: use deterministic training
    :var fast_dev_run: use fast dev run
    :var detect_anomaly: use detect anomaly
    :var accelerator: use ['gpu', 'cpu']
    :var devices: devices train
    """

    min_epochs: int = 20
    max_epochs: int = 30
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 1
    gradient_clip_val: Optional[float] = 0.1
    gradient_clip_algorithm: Optional[Literal["norm", "value"]] = "norm"
    deterministic: bool = False
    fast_dev_run: bool = False
    detect_anomaly: bool = False
    accelerator: str = "gpu"
    devices: List[int] = [0]


class ExperimentConfig(_BaseValidatedConfig):
    """
    Experiment config

    :var project_name: name of project for clearml
    :var experiment_name: name of experiment in project for clearml
    :var trainer_config: trainer config obj
    :var data_config: data config obj
    :var config_full_model: model config obj
    """

    project_name: str = "BERTSummarization"
    experiment_name: str = "exp_main_pretrained_bert"
    trainer_config: TrainerConfig = Field(default=TrainerConfig())
    data_config: DataConfig = Field(default=DataConfig())
    config_full_model: ModelConfig = Field(default=ModelConfig())
