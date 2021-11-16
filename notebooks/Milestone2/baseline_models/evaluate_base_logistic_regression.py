import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression

from notebooks.Milestone2.Model import Model
from notebooks.Milestone2.baseline_models.save_plot import save_plot


def evaluate_base_logistic_regression(data: DataFrame) -> Model:
    Y = data['isGoal']
    X = data['goalDist']

    log_reg = Model(
        predictor=LogisticRegression(),
        params={},
        X=X,
        Y=Y,
        name="Model 1"
    )

    log_reg.predictor.fit(log_reg.X_train, log_reg.Y_train)
    print(f'Base Logistic Regression Model Accuracy: {log_reg.accuracy()}.')

    predictions = log_reg.predictor.predict(log_reg.X_val)
    print(f'Unique Predicted Values : {np.unique(predictions)}')
    __plot_predictions(log_reg.Y_val, log_reg.X_val.flatten(), predictions)

    log_reg.create_experiment('Q3_Model1.pkl')

    return log_reg


def __plot_predictions(actual_values, distances, predictions):
    sns.scatterplot(x=distances, y=actual_values, label='Actual Value')
    sns.scatterplot(x=distances, y=predictions, label='Predicted Value')
    plt.xlabel('Distance')
    title = 'Goal Predictions Compared to Actual Goals Based on Distance Feature'
    plt.title(title)
    save_plot(title)
    plt.show()
