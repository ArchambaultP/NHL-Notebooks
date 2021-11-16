from comet_ml import Experiment
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from dotenv import load_dotenv

load_dotenv()


class Model:

    def __init__(self, predictor: object = None, params: dict = None, X=None, Y=None, name=None, keep_model_file=False):
        if predictor is None or params is None or X is None or Y is None:
            print('Invalid Model')
            return

        self.keep_model_file = keep_model_file
        self.params = params
        self.Name = name
        self.predictor = predictor
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, stratify=Y)

        if len(X_train.shape) <= 1:
            X_train = X_train.to_numpy().reshape(-1, 1)
            X_val = X_val.to_numpy().reshape(-1, 1)

        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val

    def fit(self):
        clf = GridSearchCV(self.predictor,
                           param_grid=self.params,
                           cv=10,
                           n_jobs=-1)

        clf.fit(self.X_train, self.Y_train)
        self.predictor = clf.best_estimator_

    def accuracy(self):
        Y_hat = self.predictor.predict(self.X_val)
        n_errors = np.sum(Y_hat != self.Y_val)
        accuracy = 1 - (n_errors / Y_hat.shape[0])
        return accuracy

    def prob_goal(self):
        """ Returns the probability that a shot was a goal, Aligned with self.X_val
        """
        Y_hat = self.predictor.predict_proba(self.X_val)
        return Y_hat[:, -1]

    def create_experiment(self, file_path):

        if hasattr(self, 'exp'):
            return self.exp

        exp = Experiment(api_key=os.getenv('COMET_ML_KEY'),
                         workspace="charlescol",
                         project_name="milestone-2")

        self.fit()

        Y_hat = self.predictor.predict(self.X_val)

        conf_matrix = confusion_matrix(self.Y_val, Y_hat)
        f1 = f1_score(self.Y_val, Y_hat)
        precision = precision_score(self.Y_val, Y_hat)
        recall = recall_score(self.Y_val, Y_hat)

        metrics = {
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "accuracy": self.accuracy(),
        }

        exp.log_dataset_hash(self.X_train)
        exp.log_parameters(self.params)
        exp.log_metrics(metrics)
        exp.log_confusion_matrix(matrix=conf_matrix)

        pickle.dump(self.predictor, open(file_path, 'wb'))
        exp.log_model(self.Name, file_path)

        self.exp = exp

        if self.keep_model_file == False:
            os.remove(file_path)

        return exp
