import configparser
from pathlib import Path


def _get_paths():
    config = configparser.ConfigParser()
    config.read(Path.home() / ".icon_data.ini")
    data_dir = config.get("paths", "data_dir", fallback="D:/lloyd/datasets")
    results_dir = config.get("paths", "results_dir", fallback="./results")
    Path(results_dir).mkdir(exist_ok=True)
    return Path(data_dir), Path(results_dir)


DATASET_DIR, RESULTS_DIR = _get_paths()
