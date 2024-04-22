from pathlib import Path

import pandas as pd

from src.constantsconfigs.constants import NEED_COLUMNS


def load_dataset(path_dataset_jsonl: Path) -> pd.DataFrame:
    """
    Load dataset from jsonl to pandas dataframe
    :param path_dataset_jsonl: path to dataset
    :return: dataset data [text, sum]
    """
    with open(path_dataset_jsonl, 'r') as file_json_l:
        dataset = pd.read_json(file_json_l, lines=True)[NEED_COLUMNS]
    return dataset
