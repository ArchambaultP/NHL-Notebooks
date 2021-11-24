from notebooks.Milestone2.Q6_best_shot.train_models import get_training_dataset

class DataMaker :
    """Class able to make specific validation data for a given model name"""
    base_df = None

    @staticmethod
    def load_all_data(season:int, type:str) :
        DataMaker.base_df = get_training_dataset([type], [season])
        if season == 2019 :
            DataMaker.base_df.insert(loc=31, column='SHOOTOUT_COMPLETE', value=[0] * DataMaker.base_df.shape[0])

    @staticmethod
    def get_baseline_data( model_name:str) -> pd.DataFrame :
        dic_features = {
            "Baseline_Model_Distance" : ['goalDist'],
            "Baseline_Model_Angle" : ['angle'],
            "Baseline_Model_Angle_Distance" : ['angle', 'goalDist'],
            "Baseline_Random_Model" : ['angle', 'goalDist'],
            }
        return DataMaker.base_df[dic_features[model_name] + ['isGoal']]

    @staticmethod
    def get_advanced_data(model_name:str) -> None :
        return DataMaker.base_df[:]

    @staticmethod
    def get_xgboost_data(model_name:str) :
        if model_name == 'xgb_model_feat_sel' :
            return DataMaker.base_df[['EmptyNet', 'Period', 'goalDist', 'HIT', 'Coordinates.y', 'GIVEAWAY', 'Change_angle', 'STOP', 'OpposingPlayers', 'Time_last_event', 'isGoal']]
        elif model_name == 'xgb_model_grid' :
            return DataMaker.base_df[['Period', 'Coordinates.x', 'Coordinates.y', 'goalDist', 'angle', 'EmptyNet', 'TotalSeconds', 'Last_coordinates.x', 'Last_coordinates.y', 'Time_last_event', 'Distance_last_event', 'Rebound', 'Speed', 'TimeSincePP', 'FriendPlayers', 'OpposingPlayers', 'Change_angle', 'BLOCKED_SHOT', 'CHALLENGE', 'FACEOFF', 'GAME_OFFICIAL', 'GIVEAWAY', 'GOAL', 'HIT', 'MISSED_SHOT', 'PENALTY', 'PERIOD_END', 'PERIOD_OFFICIAL', 'PERIOD_READY', 'PERIOD_START', 'SHOOTOUT_COMPLETE', 'SHOT', 'STOP', 'TAKEAWAY', 'Backhand', 'Deflected', 'Slap Shot', 'Snap Shot', 'Tip-In', 'Wrap-around', 'Wrist Shot', 'isGoal']]