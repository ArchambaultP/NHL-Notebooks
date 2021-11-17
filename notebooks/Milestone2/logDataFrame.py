from comet_ml import Experiment
from ift6758.features import tidy_data as td
import pandas as pd
from ift6758.data import import_dataset
import json
import math
import os
from os.path import dirname, abspath
from pathlib import Path
from dotenv import load_dotenv

def get_training_dataset(GamePK = 2017021065) -> pd.DataFrame:

    df = pd.DataFrame()
    raw_data = import_dataset(2017, 'R', returnData=True)
    pbp_data = td.get_playbyplay_data(raw_data)
    pdp_tidied = td.tidy_playbyplay_data(pbp_data)
    df = td.tidy2_playbyplay_data(raw_data, pd.concat([pd.DataFrame(pbp_data),pdp_tidied], axis=1))
    df.reset_index(drop=True, inplace=True)
    df = df[df['GamePk'] == GamePK]
    df = df.drop(columns=['GamePk'])
    
    return  df
 
def save_set(file_name: str, df):
    root = Path(dirname(abspath(__file__)))
    data_dir = root / "log_dataframe"
    if not data_dir.exists():
        data_dir.mkdir()

    file_path = data_dir / file_name
    df.to_csv(file_path)
 
df = get_training_dataset(2017021065)


experiment = Experiment(
                api_key=os.getenv('COMET_API_KEY'),
                project_name='milestone-2-feature_engineering_data',
                workspace="charlescol",
)

experiment.log_dataframe_profile(
df,
name='wpg_v_wsh_2017021065', # keep this name
dataframe_format='csv' # ensure you set this flag!
)
