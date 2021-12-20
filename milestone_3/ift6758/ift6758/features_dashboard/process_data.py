import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer
import xgboost as xgb


def get_columns():
    
    columns = ['Period', 'Coordinates.x', 'Coordinates.y', 'goalDist', 'angle', 'EmptyNet',  
                   'TotalSeconds', 'Last_coordinates.x', 'Last_coordinates.y', 'Time_last_event', 
                   'Distance_last_event', 'Rebound', 'Speed', 'TimeSincePP', 'FriendPlayers', 
                   'OpposingPlayers', 'Change_angle', 'BLOCKED_SHOT', 'CHALLENGE', 'FACEOFF', 
                   'GAME_OFFICIAL', 'GIVEAWAY', 'GOAL', 'HIT', 'MISSED_SHOT', 'PENALTY', 'PERIOD_END', 
                   'PERIOD_OFFICIAL', 'PERIOD_READY', 'PERIOD_START', 'SHOOTOUT_COMPLETE', 'SHOT', 'STOP', 
                   'TAKEAWAY', 'Backhand', 'Deflected', 'Slap Shot', 'Snap Shot', 'Tip-In', 'Wrap-around', 
                   'Wrist Shot'] 
    
    return columns

def process_dataframe(df):

    df.reset_index(drop=True, inplace=True)
    try:
        df = df.drop(columns=['GamePk'])
    except Exception as e:
        df = df
    columns = get_columns()

    df["Last_event_type"] = df["Last_event_type"].astype(str)
    lb = LabelBinarizer()
    transfo = lb.fit_transform(df["Last_event_type"])
    df = df.drop(columns=['Last_event_type'])
    df[lb.classes_.astype(object)] = transfo

    df.loc[df["ShotType"].isnull(), "ShotType"] = "Slap Shot"
    transfo = lb.fit_transform(df["ShotType"])
    df = df.drop(columns=['ShotType'])
    df[lb.classes_.astype(object)] = transfo

    
    for loc, column in enumerate(columns):
        if column not in df.columns :
                df.insert(loc=loc, column=column, value=[0] * df.shape[0])
    
    Y = df['isGoal']
    df = df.loc[:, df.columns != 'isGoal']
    
    df_xgb = df.loc[:,columns]
    columns.append('team')
    df = df.loc[:,columns]
    
    dmatrix = xgb.DMatrix(data=df_xgb,label=Y)
    
    return df, dmatrix