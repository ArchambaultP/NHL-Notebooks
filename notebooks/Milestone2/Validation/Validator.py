from notebooks.Milestone2.Model import Model, Plots
from notebooks.Milestone2.common.save_plot import save_plot
from notebooks.Milestone2.Validation import DataMaker, ModelLoader

import pandas as pd
import xgboost as xgb
import numpy as np

class Validator() :
    """Class to make the validation test for a given Model object and plot it if necessary"""
    class Model_XGB(Model) :
        def __init__(self, predictor: object = None, params: dict = None, X=None, Y=None, name=None, keep_model_file=False, test_size=0.2) :
            Model.__init__(self, predictor=predictor, params=params, X=X, Y=Y, name=name, keep_model_file=keep_model_file, test_size=test_size)
            self.dtest = xgb.DMatrix(data=self.X_val, label=self.Y_val)

        def accuracy(self):
            Y_hat = [round(value) for value in self.goal_probability()]
            return 1 - (np.sum(Y_hat != self.Y_val) / self.Y_val.shape[0])

        def goal_probability(self):
            return self.predictor.predict(self.dtest)[:]

    @staticmethod
    def make_validation_model(sk_model:object, data:pd.DataFrame, name:str, Model_class=Model) -> object :
        Y = data['isGoal']
        X = data[data.columns.drop('isGoal')]

        model = Model_class(predictor=sk_model, params={}, X=X, Y=Y, name="Validation_" + name, test_size=float(1))
        print(f'{name} accuracy: {model.accuracy()}.')

        return model

    @staticmethod
    def plot_validation_results(models:list, name:str) -> None :
        plot = Plots(models)
        plot.save_and_show_plots(name, dirname(abspath(__file__)))

def validate(season:int, type:str) -> None :
    """ Method to start validation worflow with all registred models on comet. Work with Regular 2019 adding SHOOTOUT_COMPLETE column with zero values.!! 
    \nInput : season -> season for test values || type -> fame type
    \nOuput : None
    \n!! WARNING : Final models work with to much features to work with Playoffs 2019 (some columns have to be added with 0 value in DataMaker) !!
    """
    DataMaker.load_all_data(season, type) # Load all data from (type,season)

    model_loader = ModelLoader()
    model_loader.get_models_from_api()
    sk_models = model_loader.get_models_from_file()

    validation_models = {'xgb': [], 'Baseline' : [], 'Final' : [] }
    for sk_model in sk_models :
        validation_data = None
        model_name = sk_model[ModelLoader.Model_Enum.NAME]
        if model_name[:3] == 'xgb' :
            validation_data = DataMaker.get_xgboost_data(model_name)
            validation_models['xgb'].append(Validator.make_validation_model(sk_model[ModelLoader.Model_Enum.MODEL], validation_data, model_name, Validator.Model_XGB))
        else : 
            if model_name[:8] == 'Baseline' :
                validation_data = DataMaker.get_baseline_data(model_name)
                validation_models['Baseline'].append(Validator.make_validation_model(sk_model[ModelLoader.Model_Enum.MODEL], validation_data, model_name))

            elif  model_name[:5] == 'Final' :
                validation_data = DataMaker.get_advanced_data(model_name)
                validation_models['Final'].append(Validator.make_validation_model(sk_model[ModelLoader.Model_Enum.MODEL], validation_data, model_name))

    for k,v in validation_models.items() :
        Validator.plot_validation_results(v, f'Validation_{season}_{type}_{k}')
