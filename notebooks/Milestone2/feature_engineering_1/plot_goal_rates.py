import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame

from notebooks.Milestone2.feature_engineering_1.save_plot import save_plot


def plot_goal_rates_per_distances_and_angles(dataset: DataFrame):
    __plot_goal_rates_per_distance(dataset)
    __plot_goal_rates_per_angle(dataset)


def __plot_goal_rates_per_distance(dataset: DataFrame):
    __plot_goal_rates(dataset, bin_width=5, col='goalDist', feature_name='Distance')


def __plot_goal_rates_per_angle(dataset: DataFrame):
    __plot_goal_rates(dataset, bin_width=2, col='angle', feature_name='Angle')


def __plot_goal_rates(dataset: DataFrame, bin_width: int, col: str, feature_name: str):
    values = dataset[col].to_numpy()
    max_value = np.amax(values)
    min_value = np.amin(values)
    rounded_min_value = int(np.floor(min_value))
    bins = range(rounded_min_value - (abs(rounded_min_value) % bin_width), int(max_value + bin_width), bin_width)

    goals = dataset[dataset['isGoal'] == 1]
    goal_hist, goal_bin_edges = np.histogram(goals[col], bins=bins)
    hist, bin_edges = np.histogram(dataset[col], bins=bins)

    goal_ratios = np.divide(goal_hist.astype(float), hist.astype(float), out=np.zeros_like(goal_hist, dtype=float),
                            where=hist != 0)

    plt.bar(bin_edges[:-1], goal_ratios, width=bin_width, align='edge', edgecolor='black')
    plt.xlim(min(bin_edges), max(bin_edges))
    title = f'Goal Rates per Shot {feature_name}'
    plt.title(title)
    plt.ylabel('Goal Rates')
    plt.xlabel(f'Shot {feature_name}')
    save_plot(title)
    plt.show()
