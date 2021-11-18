from os.path import dirname, abspath

import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame

from notebooks.Milestone2.common.save_plot import save_plot


def plot_shots_per_distances_and_angles(dataset: DataFrame):
    __plot_shot_counts_binned_by_distance(dataset)
    __plot_shot_counts_binned_by_angle(dataset)
    __plot_shots_by_angle_and_distance(dataset)


def __plot_shot_counts_binned_by_distance(dataset: DataFrame):
    sns.histplot(dataset[['isGoal', 'goalDist']], x='goalDist', hue='isGoal')
    title = 'Shot Count per Distance'
    plt.title(title)
    plt.ylabel('Shot Count')
    plt.xlabel('Shot Distance')
    __save_plot(title)
    plt.show()


def __plot_shot_counts_binned_by_angle(dataset: DataFrame):
    sns.histplot(dataset[['isGoal', 'angle']], x='angle', hue='isGoal')
    title = 'Shot Count per Angle'
    plt.title(title)
    plt.ylabel('Shot Count')
    plt.xlabel('Shot Angle')
    __save_plot(title)
    plt.show()


def __plot_shots_by_angle_and_distance(dataset: DataFrame):
    sns.jointplot(x=dataset.goalDist, y=dataset.angle)
    title = 'Shots in Distance-Angle Feature Space'
    plt.suptitle(title)
    plt.xlabel('Shot Distance')
    plt.ylabel('Shot Angle')
    __save_plot(title)
    plt.show()


def __save_plot(title: str):
    save_plot(title, dirname(abspath(__file__)))
