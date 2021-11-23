from typing import List

from comet_ml import Experiment
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import cv
from dotenv import load_dotenv

load_dotenv()


class Model_XGB:

    def __init__(self, params: {'objective':'binary:logistic'}, num_boost_round=5, X=None, Y=None, name=None, keep_model_file=False):
        if X is None or Y is None or params is None:
            print('Invalid Model')
            return

        self.keep_model_file = keep_model_file
        self.params = params
        self.num_boost_round = num_boost_round
        self.Name = name
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, stratify=Y)

        if len(X_train.shape) <= 1:
            X_train = X_train.to_numpy().reshape(-1, 1)
            X_val = X_val.to_numpy().reshape(-1, 1)

        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val
        
        self.dtrain = xgb.DMatrix(data=X_train,label=Y_train)
        self.dtest = xgb.DMatrix(data=X_val,label=Y_val)

    def fit(self):
        self.xg_class = xgb.train(params = self.params, dtrain=self.dtrain, num_boost_round=self.num_boost_round)

    def goal_probability(self, data=" "):
        if data == " ":
            Y_hat = self.xg_class.predict(self.dtest)
        else:
            Y_hat = self.xg_class.predict(data)
        return Y_hat
    
    def predict(self, data):
        Y_hat = self.goal_probability(data)
        Y_hat = [round(value) for value in Y_hat]
        return Y_hat
    
    def accuracy(self):
        Y_hat = self.predict(self.dtest)
        n_errors = np.sum(np.array(Y_hat) != self.Y_val)
        accuracy = 1 - (n_errors / self.Y_val.shape[0])
        return accuracy
    
    def grid_search_fit(self):
        
        params = {}

        num_boost_round = 100
        
        gridsearch_params = [
            (max_depth, min_child_weight, eta)
            for max_depth in range(2,10)
            for min_child_weight in range(1,3)
            for eta in [0.2, 0.3, 0.4, 0.5, 0.8, 1]
        ]
        min_error = float("Inf")
        best_params = None
        for max_depth, min_child_weight,eta in gridsearch_params:
            params['max_depth'] = max_depth
            params['min_child_weight'] = min_child_weight
            params['eta'] = eta
            params['objective'] = 'binary:logistic'
            # Run CV
            cv_results = xgb.cv(
                params,
                self.dtrain,
                num_boost_round=num_boost_round,
                seed=123,
                nfold=5,
                metrics={'error'},
                early_stopping_rounds=10
            )
            mean_error = cv_results['test-error-mean'].min()
            boost_rounds = cv_results['test-error-mean'].argmin()
            if mean_error < min_error:
                min_error = mean_error
                best_params = params.copy()
                best_b_round = boost_rounds 
        #print('Best error: ' + str(max_error) +' with params: ' + str(best_params) + ' and num_boost_round = ' + str(best_b_round))
        self.params, self.num_boost_round = best_params, best_b_round 
        self.fit() 
    
    def create_experiment(self, file_path, tags: List[str] = None):

        if hasattr(self, 'exp'):
            return self.exp

        exp = Experiment(api_key=os.getenv('COMET_ML_KEY'),
                         workspace="charlescol",
                         project_name="milestone-2")

        Y_hat = np.array(self.predict(self.dtest))

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

        if tags is None:
            tags = []
        if self.Name is not None:
            tags = tags + [self.Name]
        exp.add_tags(tags)

        pickle.dump(self.xg_class, open(file_path, 'wb'))
        exp.log_model(self.Name, file_path)

        self.exp = exp

        if self.keep_model_file == False:
            os.remove(file_path)

        return exp
   