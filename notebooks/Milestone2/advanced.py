import comet_ml

from pandas import DataFrame
import pandas as pd
from ift6758.data import import_dataset
from ift6758.features import tidy_data as td

from sklearn.feature_extraction import DictVectorizer

import importlib
import notebooks
importlib.reload(notebooks)

from notebooks.Milestone2.advanced_models import evaluate_xgb
from notebooks.Milestone2.advanced_models import feature_selec_xgb
from notebooks.Milestone2.Model import Model, Plots
from notebooks.Milestone2.Model_XGB import Model_XGB

from dotenv import load_dotenv
load_dotenv()

def one_hot(frame, feat, letter):
    df = frame.copy()
    df_cat = df[[feat]].copy()
    df = df.loc[:, df.columns != feat].copy()
       
    v = DictVectorizer()
    df_cat = pd.DataFrame(v.fit_transform(df_cat.to_dict('records')).toarray())
    columns = []
    for i in range (df_cat.shape[1]):
      columns.append(letter+str(i))
    df_cat.columns = columns
    
    df_cat = df_cat.astype(int)
    df = pd.concat([df, df_cat], axis=1) 
    return df


def get_training_dataset() -> DataFrame:
    train_split_seasons = [2015, 2016, 2017, 2018]
    #train_split_seasons = [2016]
    training_dataset = DataFrame()
    subset = DataFrame()
    for s in train_split_seasons:
        for game_type in ['R','P']:
        #for game_type in ['P']:
            raw_data = import_dataset(s, game_type, returnData=True)
            pbp_data = td.get_playbyplay_data(raw_data)
            pdp_tidied = td.tidy_playbyplay_data(pbp_data)
            pdp2_tidied = td.tidy2_playbyplay_data(raw_data, pd.concat([pd.DataFrame(pbp_data),pdp_tidied], axis=1))
            training_dataset = training_dataset.append(pdp2_tidied)
            training_dataset.reset_index(drop=True, inplace=True)
    
    training_dataset = training_dataset.drop(columns=['GamePk'])
    training_dataset = training_dataset.dropna()
    training_dataset.reset_index(drop=True, inplace=True)
    training_dataset = one_hot(training_dataset, 'ShotType', 'f')
    training_dataset = one_hot(training_dataset, 'Last_event_type', 'g')
    
    return  training_dataset


training_dataset = get_training_dataset()
evaluate_xgb(training_dataset)












