import pandas as pd
from pathlib import Path
from constantsconfigs.constants import NEED_COLUMNS, DEFAULT_PROJECT_PATH


def load_dataset(path_dataset_jsonl: Path) -> pd.DataFrame:
    """
    Load dataset from jsonl to pandas dataframe
    :param path_dataset_jsonl: path to dataset
    :return: dataset data [text, sum]
    """
    with open(DEFAULT_PROJECT_PATH / path_dataset_jsonl, "r") as file_json_l:
        dataset = pd.read_json(file_json_l, lines=True)[list(NEED_COLUMNS)]
    return dataset
