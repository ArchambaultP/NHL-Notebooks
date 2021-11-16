from os.path import dirname, abspath
from pathlib import Path

from matplotlib import pyplot as plt


def save_plot(file_name: str):
    root = Path(dirname(abspath(__file__)))
    data_dir = root / "plots"
    if not data_dir.exists():
        data_dir.mkdir()

    file_path = data_dir / file_name
    plt.savefig(file_path)
