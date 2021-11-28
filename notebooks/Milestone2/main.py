import comet_ml
from pandas import DataFrame

from ift6758.data import import_dataset
from ift6758.features import tidy_data as td
from notebooks.Milestone2.baseline_models import evaluate_logistic_classifiers
from notebooks.Milestone2.feature_engineering_1 import plot_shots_per_distances_and_angles, \
    plot_goal_rates_per_distances_and_angles, plot_goal_counts_binned_by_distance
from notebooks.Milestone2.Q6_best_shot import train_models

def main():
    training_dataset = get_training_dataset()

    # Feature Engineering 1
    plot_shots_per_distances_and_angles(training_dataset)
    plot_goal_rates_per_distances_and_angles(training_dataset)
    plot_goal_counts_binned_by_distance(training_dataset)

    # Baseline Models
    evaluate_logistic_classifiers(training_dataset)

    # Various Model Attemps
    train_models()
    return

def get_training_dataset() -> DataFrame:
    train_split_seasons = [2015, 2016, 2017, 2018]
    training_dataset = DataFrame()
    for s in train_split_seasons:
        raw_data = import_dataset(s, "P", returnData=True)
        pbp_data = td.get_playbyplay_data(raw_data)
        pdp_tidied = td.tidy_playbyplay_data(pbp_data)
        training_dataset = training_dataset.append(pdp_tidied)

    return training_dataset.reset_index()

if __name__ == "__main__":
    main()
    #train_models()
