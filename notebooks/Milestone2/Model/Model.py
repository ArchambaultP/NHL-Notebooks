import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Model():

    def __init__(self, predictor=None, X=None, Y=None, name=None):
        if predictor is None or X is None or Y is None:
            print('Invalid Model')
            return
        
        self.Name = name
        self.predictor = predictor
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, stratify=Y)

        if len(X_train.shape) <= 1:
            X_train = X_train.to_numpy().reshape(-1,1)
            X_val = X_val.to_numpy().reshape(-1,1)

        self.X_train = X_train
        self.X_val = X_val
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.fit()
    
    def fit(self):
        self.predictor.fit(self.X_train, self.Y_train)
    
    def accuracy(self):
        Y_hat = self.predictor.predict(self.X_val)
        n_errors = np.sum(Y_hat != self.Y_val)
        accuracy = 1 - (n_errors/Y_hat.shape[0])
        return accuracy
    
    def prob_goal(self):
        """ Returns the probability that a shot was a goal, Aligned with self.X_val
        """
        Y_hat = self.predictor.predict_proba(self.X_val)
        return Y_hat[:,-1]

