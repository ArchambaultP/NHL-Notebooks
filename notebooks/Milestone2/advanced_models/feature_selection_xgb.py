from os.path import dirname, abspath
import numpy as np
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
import shap
from notebooks.Milestone2.Model_XGB import Model_XGB


def feat_sel_get_score(model,n_feat):
    # Feature Importance Get_score gain
    feat_imp = model.get_score(importance_type='gain')
    feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
    columns_getscore = list(feat_imp)[0:n_feat]
    return columns_getscore


def feat_sel_shap(model, X, Y, n_feat):
    # Feature selection SHAP
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, stratify=Y)
    model.fit(X_train,Y_train)
    columns = X.columns.tolist() 
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([columns, shap_sum.tolist()]).T
    importance_df.columns = ['column_name', 'shap_importance']
    importance_df = importance_df.sort_values('shap_importance', ascending=False)
    columns_sharp =  importance_df.iloc[0:n_feat,0].values.tolist()
    return columns_sharp