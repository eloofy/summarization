import yaml
from typing import Union
from pathlib import Path
from constantsconfigs.configs import ExperimentConfig


def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """
    Load file from yaml.

    :param config_path: path to yaml file
    :return: ExperimentConfig obj
    """
    config_path = Path(config_path)
    with open(config_path, "r") as file_cfg:
        config_dict = yaml.safe_load(file_cfg)
    config_dict["_base_path"] = config_path.parent
    return ExperimentConfig(**config_dict)
