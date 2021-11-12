import pandas as pd

from ift6758.data import import_dataset
from ift6758.features import tidy_data as td
from notebooks.Milestone2.feature_engineering_1 import plot_shots_per_distances_and_angles


def main():
    training_dataset = get_training_dataset()

    # Feature Engineering 1
    plot_shots_per_distances_and_angles(training_dataset)


def get_training_dataset() -> pd.DataFrame:
    train_split_seasons = [2015, 2016, 2017, 2018]
    training_dataset = pd.DataFrame()
    for s in train_split_seasons:
        raw_data = import_dataset(s, "P", returnData=True)
        pbp_data = td.get_playbyplay_data(raw_data)
        pdp_tidied = td.tidy_playbyplay_data(pbp_data)
        training_dataset = training_dataset.append(pdp_tidied)

    return training_dataset.reset_index()


if __name__ == "__main__":
    main()