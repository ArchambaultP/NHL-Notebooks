import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame

from notebooks.Milestone2.feature_engineering_1.save_plot import save_plot


def plot_shots_per_distances_and_angles(dataset: DataFrame):
    __plot_shot_counts_binned_by_distance(dataset)
    __plot_shot_counts_binned_by_angle(dataset)
    __plot_shots_by_angle_and_distance(dataset)


def __plot_shot_counts_binned_by_distance(dataset: DataFrame):
    sns.histplot(dataset[['isGoal', 'goalDist']], x='goalDist', hue='isGoal')
    title = 'Shot Count per Distance'
    plt.title(title)
    plt.xlabel('Shot Distance')
    save_plot(title)
    plt.show()


def __plot_shot_counts_binned_by_angle(dataset: DataFrame):
    sns.histplot(dataset[['isGoal', 'angle']], x='angle', hue='isGoal')
    title = 'Shot Count per Angle'
    plt.title(title)
    plt.xlabel('Shot Angle')
    save_plot(title)
    plt.show()


def __plot_shots_by_angle_and_distance(dataset: DataFrame):
    sns.jointplot(x=dataset.goalDist, y=dataset.angle)
    title = 'Shots in Distance-Angle Feature Space'
    plt.suptitle(title)
    plt.xlabel('Shot Distance')
    plt.ylabel('Shot Angle')
    save_plot(title)
    plt.show()



