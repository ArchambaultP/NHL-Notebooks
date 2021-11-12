from os.path import dirname, abspath
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_shots_per_distances_and_angles(training_dataset: pd.DataFrame):
    __plot_shot_counts_binned_by_distance(training_dataset)
    __plot_shot_counts_binned_by_angle(training_dataset)
    __plot_shots_by_angle_and_distance(training_dataset)


def __plot_shots_by_angle_and_distance(training_dataset: pd.DataFrame):
    sns.jointplot(x=training_dataset.goalDist, y=training_dataset.angle)
    title = 'Shots in Distance-Angle Feature Space'
    plt.suptitle(title)
    plt.xlabel('Shot Distance')
    plt.ylabel('Shot Angle')
    __save_plot(title)
    plt.show()


def __plot_shot_counts_binned_by_angle(training_dataset: pd.DataFrame):
    sns.histplot(training_dataset[['isGoal', 'angle']], x='angle', hue='isGoal')
    title = 'Shot Count per Angle'
    plt.title(title)
    plt.xlabel('Shot Angle')
    __save_plot(title)
    plt.show()


def __plot_shot_counts_binned_by_distance(training_dataset: pd.DataFrame):
    sns.histplot(training_dataset[['isGoal', 'goalDist']], x='goalDist', hue='isGoal')
    title = 'Shot Count per Distance'
    plt.title(title)
    plt.xlabel('Shot Distance')
    __save_plot(title)
    plt.show()


def __save_plot(file_name: str):
    root = Path(dirname(abspath(__file__)))
    data_dir = root / "plots"
    if not data_dir.exists():
        data_dir.mkdir()

    file_path = data_dir / file_name
    plt.savefig(file_path)
