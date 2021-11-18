from pathlib import Path

from matplotlib import pyplot as plt


def save_plot(file_name: str, directory: str):
    root = Path(directory)
    data_dir = root / "plots"
    if not data_dir.exists():
        data_dir.mkdir()

    file_path = data_dir / file_name
    plt.savefig(file_path)
