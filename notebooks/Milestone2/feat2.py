import sys
sys.path.append('../../')
from ift6758.features import tidy_data as td
import pandas as pd
from ift6758.data import import_dataset
import json
import math
from notebooks.Milestone2.feature_engineering_2 import save_set


def get_training_dataset() -> pd.DataFrame:
    train_split_seasons = [2015, 2016, 2017, 2018]
    training_dataset = pd.DataFrame()
    subset = pd.DataFrame()
    for s in train_split_seasons:
        for game_type in ['R','P']:
            raw_data = import_dataset(s, game_type, returnData=True)
            #raw_data = json.load(open("../ift6758/data/dataset/"+str(s)+"_regular.json",))
            pbp_data = td.get_playbyplay_data(raw_data)
            pdp_tidied = td.tidy_playbyplay_data(pbp_data)
            pdp2_tidied = td.tidy2_playbyplay_data(raw_data, pd.concat([pd.DataFrame(pbp_data),pdp_tidied], axis=1))
            training_dataset = training_dataset.append(pdp2_tidied)
            training_dataset.reset_index(drop=True, inplace=True)
    
    subset = training_dataset[training_dataset['GamePk'] == 2017021065]
    training_dataset = training_dataset.drop(columns=['GamePk'])
    subset = subset.drop(columns=['GamePk'])
    
    return  training_dataset, subset

training_dataset, subset = get_training_dataset()

save_set.save_set('wpg_v_wsh_2017021065.csv', subset)
save_set.save_set('training_set.csv', training_dataset)
