from os.path import dirname, abspath

from pandas import DataFrame
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from notebooks.Milestone2.Model import Model, Plots
from notebooks.Milestone2.baseline_models.evaluate_base_logistic_regression import \
    evaluate_base_logistic_regression_with_distance_feature


def evaluate_logistic_classifiers(data: DataFrame):
    baseline_models_tag = 'Baseline Models'
    distance_logistic_regression = evaluate_base_logistic_regression_with_distance_feature(data)

    Y = data['isGoal']

    angle_logistic_regression = Model(
        predictor=LogisticRegression(),
        params={},
        X=data['angle'],
        Y=Y,
        name="Angle Logistic Regression"
    )
    angle_logistic_regression.fit()
    angle_logistic_regression.create_experiment('Baseline_Model_Angle.pkl', [baseline_models_tag])

    dist_angle_log_reg = Model(
        predictor=LogisticRegression(),
        params={},
        X=data[['goalDist', 'angle']],
        Y=Y,
        name="Distance-Angle Logistic Regression"
    )
    dist_angle_log_reg.fit()
    dist_angle_log_reg.create_experiment('Baseline_Model_Angle_Distance.pkl', [baseline_models_tag])

    random_baseline = Model(
        predictor=DummyClassifier(strategy='uniform'),
        params={},
        X=data[['goalDist', 'angle']],
        Y=Y,
        name="Uniform Random Baseline"
    )
    random_baseline.fit()
    random_baseline.create_experiment('Baseline_Random_Model.pkl', [baseline_models_tag])

    plot = Plots([distance_logistic_regression, angle_logistic_regression, dist_angle_log_reg, random_baseline])
    plot.save_and_show_plots(baseline_models_tag, dirname(abspath(__file__)))
