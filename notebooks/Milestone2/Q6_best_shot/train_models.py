
from ift6758.features import tidy_data as td
import pandas as pd
from ift6758.data import import_dataset
from notebooks.Milestone2.Model import Model, Plots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.kernel_ridge import KernelRidge
import numpy as np

def train_models():

    data = get_training_dataset("P")
    Y = data['isGoal']
    X = data[data.columns.drop('isGoal')]

    # train_kernel_ridge(X,Y)
    nb = train_NB(X,Y)
    rf = train_random_forest(X,Y)

    plot = Plots([nb,rf])
    plot.show_plots()
    # nb.create_experiment('Test_NB.pkl', ['Naive Bayes Q6'])
    return

def train_random_forest(X, Y):
    print("Training Random Forest")
    rf_params = {"n_estimators":[1,10,20,40,100], "bootstrap":[True, False], "max_depth":[1,2,4,8], "criterion":['gini', 'entropy']}
    clf_random_forest = Model(
        predictor=RandomForestClassifier(),
        params=rf_params,
        X=X,
        Y=Y,
        name="Random Forest Classifier"
    )
    clf_random_forest.random_search_fit()
    print(f"Random Forest Accuracy : {clf_random_forest.accuracy()}")
    return clf_random_forest

def train_svm(X,Y):
    print("Training SVM")
    svm_params = {'max_iter':[5000], "kernel":["linear", "poly", "rbf", "sigmoid"]}
    clf_svm = Model(
        predictor=SVC(),
        params=svm_params,
        X=X,
        Y=Y,
        name="SVM Classifier"
    )
    clf_svm.random_search_fit()
    print(f"SVM Accuracy : {clf_svm.accuracy()}")
    return clf_svm

def train_NB(X,Y):
    print("Training Naive Bayes")
    clf_NB = Model(
        predictor=GaussianNB(),
        params={},
        X=X,
        Y=Y,
        name="Naive Bayes Classifier"
    )
    clf_NB.random_search_fit()
    print(f"Naive Bayes Accuracy : {clf_NB.accuracy()}")
    return clf_NB


def train_kernel_ridge(X,Y):
    print("Training Kernel Ridge")
    kr_params = {'alpha':[0, 0.001, 0.01, 0.1, 0.5, 1, 2], "kernel":['chi2', 'linear', 'poly', 'rbf', 'laplacian', 'sigmpoid']}
    
    clf_kr = Model(
        predictor=KernelRidge(),
        params=kr_params,
        X=X,
        Y=Y,
        name="Kernel Ridge Classifier"
    )
    clf_kr.random_search_fit()
    print(f"Kernel Ridge Accuracy : {clf_kr.accuracy()}")
    return clf_kr

def get_training_dataset(gt : list  = None) -> pd.DataFrame:
    train_split_seasons = [2015]#[2015, 2016, 2017, 2018]
    training_dataset = pd.DataFrame()
    subset = pd.DataFrame()

    if gt is None:
        gt = ['R', 'P']

    for s in train_split_seasons:
        for game_type in gt:
            raw_data = import_dataset(s, game_type, returnData=True)
            pbp_data = td.get_playbyplay_data(raw_data)
            pdp_tidied = td.tidy_playbyplay_data(pbp_data)
            pdp2_tidied = td.tidy2_playbyplay_data(raw_data, pd.concat([pd.DataFrame(pbp_data),pdp_tidied], axis=1))
            training_dataset = training_dataset.append(pdp2_tidied)
            training_dataset.reset_index(drop=True, inplace=True)
    
    training_dataset = training_dataset.drop(columns=['GamePk'])

    lb = LabelBinarizer()
    transfo = lb.fit_transform(training_dataset["Last_event_type"])
    training_dataset = training_dataset.drop(columns=['Last_event_type'])
    training_dataset[lb.classes_.astype(object)] = transfo

    training_dataset.loc[training_dataset["ShotType"].isnull(), "ShotType"] = "Slap Shot"
    transfo = lb.fit_transform(training_dataset["ShotType"])
    training_dataset = training_dataset.drop(columns=['ShotType'])
    training_dataset[lb.classes_.astype(object)] = transfo
    
    return  training_dataset