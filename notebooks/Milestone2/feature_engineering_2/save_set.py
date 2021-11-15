from os.path import dirname, abspath
from pathlib import Path

from matplotlib import pyplot as plt


def save_set(file_name: str, df):
    root = Path(dirname(abspath(__file__)))
    data_dir = root / "feat2_dataset"
    if not data_dir.exists():
        data_dir.mkdir()

    file_path = data_dir / file_name
    df.to_csv(file_path)