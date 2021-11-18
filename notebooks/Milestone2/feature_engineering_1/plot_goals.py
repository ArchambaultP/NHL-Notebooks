from os.path import dirname, abspath

import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame

from notebooks.Milestone2.common.save_plot import save_plot


def plot_goal_counts_binned_by_distance(dataset: DataFrame):
    goals = dataset[dataset['isGoal'] == 1]
    sns.histplot(goals[['EmptyNet', 'goalDist']], x='goalDist', hue='EmptyNet')
    title = 'Goal Count per Distance'
    plt.title(title)
    plt.ylabel('Goal Count')
    plt.xlabel('Shot Distance')
    save_plot(title, dirname(abspath(__file__)))
    plt.show()
