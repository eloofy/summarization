from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field, model_validator


class _BaseValidatedConfig(BaseModel):
    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True


class SerializableOBj(_BaseValidatedConfig):
    """
    SerializableOBj cfg for import
    """

    target_class: str
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class EncoderTokenizer(_BaseValidatedConfig):
    """
    TokenizerConfig cfg for encoder

    :var load_path_tokenizer: path to tokenizer
    :var special_tokens: special tokens add
    :var encoder_max_length: max length of encoder
    :var encoder_padding: padding
    :var encoder_truncate: encoder truncate
    """

    load_path_tokenizer: str
    special_tokens: Optional[Dict[str, str]]
    encoder_max_length: int
    encoder_padding: str
    encoder_truncate: bool
    padding_side: Optional[str]


class DecoderTokenizer(_BaseValidatedConfig):
    """
    TokenizerConfig cfg for decoder

    :var load_path_tokenizer: path to tokenizer
    :var special_tokens: special tokens add
    :var decoder_max_length: max length of decoder
    :var decoder_padding: padding
    :var decoder_truncate: decoder truncate
    """

    load_path_tokenizer: str
    special_tokens: Optional[Dict[str, str | List]]
    decoder_max_length: int
    decoder_padding: str
    decoder_truncate: bool
    padding_side: Optional[str]


class EncoderConfig(_BaseValidatedConfig):
    """
    Encoder config

    :var bert_base_model_params: base params (no pretrained)
    :var pretrained: is pretrained use
    :var name_encoder_pretrained_model: name of pretrained model for encoder
    :var weights_pretrained_path: load weights from local path project
    :var freeze_layers: dict for freeze layers
    :var decoder_hid_size: dec hid size

    """

    bert_base_model_params: Optional[Dict[str, Any]]
    pretrained: bool
    name_encoder_pretrained_model: str
    weights_pretrained_path: Path
    freeze_layers: Optional[List]
    decoder_hid_size: int


class DecoderConfig(_BaseValidatedConfig):
    """
    Encoder config

    :var gpt2_base_model_params: base params (no pretrained)
    :var pretrained: is pretrained use
    :var name_decoder_pretrained_model: name of pretrained model for decoder
    :var weights_pretrained_path: load weights from local path project
    :var padding_side: side of padding
    :var add_cross_attentions: whatever use EncoderDecoder
    :var freeze_layers: dict for freeze layers
    """

    gpt2_base_model_params: Optional[Dict[str, Any]]
    pretrained: bool
    name_decoder_pretrained_model: Optional[Path]
    weights_pretrained_path: Optional[Path]
    padding_side: str
    add_cross_attentions: bool
    freeze_layers: Optional[List]


class DataConfig(_BaseValidatedConfig):
    """
    Data config

    :var batch_size: batch size
    :var pin_memory: pin memory
    :var shuffle: shuffle data
    :var dataset_name_json_train_first: path to train dataset json first
    :var dataset_name_json_train_second: path to train dataset json second
    :var dataset_test_val: path to val dataset
    :var encoder_tokenizer: encoder tokenizer
    :var decoder_tokenizer: decoder tokenizer
    :var num_samples_train: number of training samples
    :var num_samples_val: number of validation samples
    :var num_workers: number of workers
    """

    batch_size: int
    pin_memory: bool
    shuffle: bool
    dataset_name_json_train_first: Path
    dataset_name_json_train_second: Path
    dataset_test_val: Path
    encoder_tokenizer: Optional[EncoderTokenizer]
    decoder_tokenizer: Optional[DecoderTokenizer]
    num_samples_train: int
    num_samples_val: int
    num_workers: int


class ModelConfig(_BaseValidatedConfig):
    """
    Model config full

    :var name_model_full: base model name
    :var encoder_model: encoder model object
    :var decoder_model: decoder model object
    :var optimizer: optimizer object
    :var set_special_tokens_to_model: set special tokens to main model
    :var scheduler: scheduler object
    :var num_warmup_steps: number of warmup steps
    :var scheduler_interval: step or epoch
    :var scheduler_frequency: frequency of step (optional)
    """

    name_model_full: str
    encoder_model: Optional[EncoderConfig]
    decoder_model: DecoderConfig
    optimizer: SerializableOBj
    set_special_tokens_to_model: bool
    scheduler: SerializableOBj
    num_warmup_steps: int
    scheduler_interval: str
    scheduler_frequency: int


class TrainerConfig(_BaseValidatedConfig):
    """
    Trainer config

    :var min_epochs: min trained epochs until use patience
    :var max_epochs: max trained epochs
    :var check_val_every_n_epoch: check val every n epochs
    :var accumulate_grad_batches: accumulate gradients over multiple batches
    :var precision: calculates precision
    :var log_every_n_steps: log results every n steps
    :var gradient_clip_val: clipping value for gradient clipping
    :var deterministic: use deterministic training
    :var fast_dev_run: use fast dev run
    :var detect_anomaly: use detect anomaly
    :var accelerator: use ['gpu', 'cpu'
    :var devices: devices train
    """

    min_epochs: int
    max_epochs: int
    check_val_every_n_epoch: int
    accumulate_grad_batches: Optional[int]
    precision: Optional[int]
    log_every_n_steps: int
    gradient_clip_val: float
    gradient_clip_algorithm: str
    deterministic: bool
    fast_dev_run: bool
    detect_anomaly: bool
    accelerator: str
    devices: List[int]


class DebugSamplesConfig(_BaseValidatedConfig):
    """
    Debug samples config

    :var each_step: each step save txt text generate
    :var data_test_path: path to test data jsonl

    """

    each_step: int
    data_test_path: Path


class Metrics(_BaseValidatedConfig):
    """
    Metrics
    """

    rouge_types: List[str]
    bert: bool
    lang: str
    verbose: bool


class ExperimentConfig(_BaseValidatedConfig):
    """
    Experiment config

    :var project_name: name of project for clearml
    :var experiment_name: name of experiment in project for clearml
    :var task: type of architecture model
    :var trainer_config: trainer config obj
    :var data_config: data config obj
    :var debug_samples_config: model config obj
    """

    project_name: str
    experiment_name: str
    task: str
    trainer_config: TrainerConfig
    data_config: DataConfig
    pr_model_config: ModelConfig
    debug_samples_config: DebugSamplesConfig
    metrics: Metrics

    @model_validator(mode="before")
    def load_sub_configs(cls, values_cfg_input):  # noqa: WPS210
        base_path = values_cfg_input.get("_base_path", Path("."))
        values_cfg_input.pop("_base_path")
        for field, path in values_cfg_input.items():
            if isinstance(path, str) and path.endswith(".yaml"):
                full_path = base_path / path
                with open(full_path, "r") as f:
                    values_cfg_input[field] = yaml.safe_load(f)
            elif isinstance(path, list):
                models = []
                for model_path in path:
                    full_path = base_path / model_path
                    with open(full_path, "r") as file_full_path:
                        models.append(yaml.safe_load(file_full_path))  # noqa: WPS220
                values_cfg_input[field] = models
        return values_cfg_input