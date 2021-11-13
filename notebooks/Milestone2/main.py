import pandas as pd
from ift6758.data import import_dataset
from ift6758.features import tidy_data as td
from notebooks.Milestone2.feature_engineering_1 import plot_shots_per_distances_and_angles, \
    plot_goal_rates_per_distances_and_angles, plot_goal_counts_binned_by_distance
from sklearn.linear_model import LogisticRegression
from Model import Model, Plots


def main():
    training_dataset = get_training_dataset()
    plot_goal_rates_per_distances_and_angles(training_dataset)
    plot_goal_counts_binned_by_distance(training_dataset)
    
    plot_logistic_q3(training_dataset)
    return


def get_training_dataset() -> pd.DataFrame:
    train_split_seasons = [2015, 2016, 2017, 2018]
    training_dataset = pd.DataFrame()
    for s in train_split_seasons:
        raw_data = import_dataset(s, "P", returnData=True)
        pbp_data = td.get_playbyplay_data(raw_data)
        pdp_tidied = td.tidy_playbyplay_data(pbp_data)
        training_dataset = training_dataset.append(pdp_tidied)

    return training_dataset.reset_index()

def plot_logistic_q3(data):
    Y = data['isGoal']

    X1 = data['goalDist']
    lr1 = LogisticRegression()
    lr1 = Model(lr1, X1, Y, name="Model 1")

    X2 = data['angle']
    lr2 = LogisticRegression()
    lr2 = Model(lr2, X2, Y, name="Model 2")

    X3 = data[['goalDist', 'angle']]
    lr3 = LogisticRegression()
    lr3 = Model(lr3, X3, Y, name="Model 3")

    plot = Plots([lr1, lr2, lr3])
    plot.plot.show()
    return


if __name__ == "__main__":
    main()
