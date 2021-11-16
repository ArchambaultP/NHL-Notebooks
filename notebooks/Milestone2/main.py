import comet_ml
import pandas as pd
from ift6758.data import import_dataset
from ift6758.features import tidy_data as td
from notebooks.Milestone2.feature_engineering_1 import plot_shots_per_distances_and_angles, \
    plot_goal_rates_per_distances_and_angles, plot_goal_counts_binned_by_distance
from sklearn.linear_model import LogisticRegression
from Model import Model, Plots


def main():
    training_dataset = get_training_dataset()

    # Feature Engineering 1
    plot_shots_per_distances_and_angles(training_dataset)
    plot_goal_rates_per_distances_and_angles(training_dataset)
    plot_goal_counts_binned_by_distance(training_dataset)

    # Baseline Models
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

    params = {'C': [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100]}

    X1 = data['goalDist']
    lr1 = LogisticRegression(**params)
    lr1 = Model(
        predictor=lr1,
        params=params,
        X=X1,
        Y=Y,
        name="Model 1"
    )
    lr1.create_experiment('Q3_Model1.pkl')

    X2 = data['angle']
    lr2 = LogisticRegression(**params)
    lr2 = Model(
        predictor=lr2,
        params=params,
        X=X2,
        Y=Y,
        name="Model 2"
    )
    lr2.fit()

    X3 = data[['goalDist', 'angle']]
    lr3 = LogisticRegression(**params)
    lr3 = Model(
        predictor=lr3,
        params=params,
        X=X3,
        Y=Y,
        name="Model 3"
    )
    lr3.fit()
    plot = Plots([lr1, lr2, lr3])
    plot.plot.show()
    return


if __name__ == "__main__":
    main()
