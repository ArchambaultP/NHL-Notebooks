from os.path import dirname, abspath
import numpy as np
from pandas import DataFrame
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
import shap
from notebooks.Milestone2.Model import Plots
from notebooks.Milestone2.Model_XGB import Model_XGB
from notebooks.Milestone2.advanced_models import feature_selection_xgb as ft

def evaluate_xgb(data: DataFrame):

    Y = data['isGoal']
    
 
    X1 = data[['goalDist', 'angle']]
    xg_base = Model_XGB(
        params={},
        X=X1,
        Y=Y,
        name="XGB base"
    )
    xg_base.fit()
    xg_base.create_experiment('xgb_base.pkl', ['xgb_model','base' ])


    X3 = data.loc[:, data.columns != 'isGoal']
    xg_advanced_grid = Model_XGB(
        params={},
        X=X3,
        Y=Y,
        name="XGB - grid search"
    )
    xg_advanced_grid.grid_search_fit()
    xg_advanced_grid.create_experiment('xgb_model_grid.pkl', ['xgb_model','complete data', 'grid_search' ])
   

    # feature selection get score
    columns = ft.feat_sel_get_score(xg_advanced_grid.xg_class, 10)
    print('get score columns selected: ')
    print(columns)
    X4 = data.loc[:, columns]
    xg_advanced_feat_score = Model_XGB(
        params={},
        X=X4,
        Y=Y,
        name="XGB - feature selection"
    )
    xg_advanced_feat_score.grid_search_fit()
    
    # feature selection shap
    columns = ft.feat_sel_shap(XGBClassifier(), X3, Y, 10)
    print('shap columns selected: ')
    print(columns)
    X5 = data.loc[:, columns]
    xg_advanced_feat_shap = Model_XGB(
        params={},
        X=X5,
        Y=Y,
        name="Teste XGB advanced - feat sel"
    )
    xg_advanced_feat_shap.grid_search_fit()
    
    
    if xg_advanced_feat_score.accuracy() > xg_advanced_feat_shap.accuracy():
        xg_feat_sel = xg_advanced_feat_score
        print('get score selected')
    else:
        xg_feat_sel = xg_advanced_feat_shap
        print('shap selected')
        
    xg_feat_sel.create_experiment('xgb_model_feat_sel.pkl', ['xgb_model','complete data', 'grid_search', 'feat sel' ])
    
    print('base acc: '+str(xg_base.accuracy()))
    print('grid acc: '+str(xg_advanced_grid.accuracy()))
    print('feat acc: '+str(xg_feat_sel.accuracy()))

    
    plot = Plots([xg_base])
    plot.save_and_show_plots('base xgb model', dirname(abspath(__file__)))
    
    plot = Plots([xg_advanced_grid])
    plot.save_and_show_plots('xgb model - complete data & grid search', dirname(abspath(__file__)))
    
    plot = Plots([xg_feat_sel])
    plot.save_and_show_plots('xgb model - feature selection', dirname(abspath(__file__)))
    
    plot = Plots([xg_base, xg_advanced_grid, xg_feat_sel])
    plot.save_and_show_plots('xgb model - feature selection', dirname(abspath(__file__)))
    

    
    
    
    


