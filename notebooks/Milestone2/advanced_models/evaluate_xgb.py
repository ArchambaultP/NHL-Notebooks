from os.path import dirname, abspath

from pandas import DataFrame
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from notebooks.Milestone2.Model import Plots

import importlib
import notebooks
importlib.reload(notebooks)
from notebooks.Milestone2.Model_XGB import Model_XGB

def evaluate_xgb(data: DataFrame):

    Y = data['isGoal']
    
 
    X1 = data[['goalDist', 'angle']]
    xg_base = Model_XGB(
        params={},
        X=X1,
        Y=Y,
        name="Teste XGB base features"
    )
    xg_base.fit()


    X2 = data.loc[:, data.columns != 'isGoal']
    xg_advanced = Model_XGB(
        params={},
        X=X2,
        Y=Y,
        name="Teste XGB advanced"
    )
    xg_advanced.fit()
    

    X3 = data.loc[:, data.columns != 'isGoal']
    xg_advanced_grid = Model_XGB(
        params={},
        X=X3,
        Y=Y,
        name="Teste XGB advanced - grid search"
    )
    xg_advanced_grid.grid_search_fit()
   
    
    X4 = data.loc[:, data.columns != 'isGoal']
    xg_advanced_feat_sel = Model_XGB(
        params={},
        X=X3,
        Y=Y,
        name="Teste XGB advanced - feat sel"
    )
    xg_advanced_feat_sel.grid_search_fit()
    
    xg_class = xg_advanced_grid.xg_class
    feat_imp = xg_class.get_score(importance_type='gain')
    feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
    df_columns = []
    for i in range(12):
        df_columns.append(list(feat_imp)[i])    
    
    X5 = X4.loc[:, df_columns[0:1]]
    xg_advanced_feat_sel_5 = Model_XGB(
        params={},
        X=X5,
        Y=Y,
        name="Teste XGB advanced - feat selection n=5"
    )
    xg_advanced_feat_sel_5.grid_search_fit()
    
    X6 = X4.loc[:, df_columns[0:8]]
    xg_advanced_feat_sel_8 = Model_XGB(
        params={},
        X=X6,
        Y=Y,
        name="Teste XGB advanced - feat selection n=8"
    )
    xg_advanced_feat_sel_8.grid_search_fit()
    
    
    X7 = X4.loc[:, df_columns[0:10]]
    xg_advanced_feat_sel_10 = Model_XGB(
        params={},
        X=X7,
        Y=Y,
        name="Teste XGB advanced - feat selection n=10"
    )
    xg_advanced_feat_sel_10.grid_search_fit()
    

    print('base acc: '+str(xg_base.accuracy()))
    print('add acc: '+str(xg_advanced.accuracy()))
    print('grid acc: '+str(xg_advanced_grid.accuracy()))
    print('grid + feat5 acc: '+str(xg_advanced_feat_sel_5.accuracy()))
    print('grid + feat8 acc: '+str(xg_advanced_feat_sel_8.accuracy()))
    print('grid + feat10 acc: '+str(xg_advanced_feat_sel_10.accuracy()))
    
    
    plot = Plots([xg_base, xg_advanced,xg_advanced_grid, xg_advanced_feat_sel_5, xg_advanced_feat_sel_8, xg_advanced_feat_sel_10])
    plot.save_and_show_plots('advanced models', dirname(abspath(__file__)))
    print('plot')
    
    #xg_base.create_experiment('advanced_models_baseData.pkl', ['advanced_model','base data' ])
    #xg_advanced.create_experiment('advanced_models_advData.pkl', ['advanced_model','advanced data' ])
    #xg_advanced_grid.create_experiment('advanced_models_advData_grid.pkl', ['advanced_model','advanced data', 'grid_search' ])
    #xg_advanced_feat_sel_5.create_experiment('advanced_models_baseData.pkl', ['advanced_model','advanced data', 'grid_search', 'feat sel 5' ])
    #xg_advanced_feat_sel_8.create_experiment('advanced_models_advData.pkl', ['advanced_model','advanced data', 'grid_search', 'feat sel 8' ])
    #xg_advanced_feat_sel_10.create_experiment('advanced_models_advData_grid.pkl', ['advanced_model','advanced data', 'grid_search', 'feat sel 10' ])

