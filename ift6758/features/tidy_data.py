import json
import pandas as pd
import re
from ift6758.ressources import local_ressource as url
from ift6758.data import fetch_boxscore_data
from dateutil import parser
from datetime import timedelta
import numpy as np

"""
We could make a function which generalizes the extraction for given parameters 
but long because of the conditions on the events and the values not included in the events 
"""
def extract_data_from_json(json) :
	"""
	Extract usefull data for a given json file
	Returns a list which contains a dictionary for each match with only the usefull data

	json : json string returned by the 'data.import_dataset' function
	"""
	data = json
	extracted_data = []
	for match in data :       
		# game_duration = parser.parse(match["gameData"]["datetime"]["endDateTime"]) - parser.parse(match["gameData"]["datetime"]["dateTime"])
		game_duration = timedelta(minutes=60)
		if len(match['liveData']['linescore']['periods']) > 3: #there is overtime
			game_duration += timedelta(minutes=5)
		for i,fact in enumerate(match['liveData']['plays']['allPlays']):
			current = {}
			event_id = fact['result']['eventTypeId']
			if  event_id == 'SHOT' or event_id == 'GOAL' : # only goals and shots
				current['GamePk'] = match['gamePk']
				current['GameDuration'] = game_duration
				current['EventTypeId'] = event_id
				current['DateTime'] = fact['about']['dateTime']
				current['PeriodTimeRemaining'] = fact['about']['periodTimeRemaining']
				current['Period'] = fact['about']['period']
				current['Team.id'] = fact['team']['id']
				current['Team.name'] = fact['team']['name']
				for player in fact['players'] :
					current[player['playerType']] = player['player']['id']
				try:
					current['Coordinates.x'] = fact['coordinates']['x']
					current['Coordinates.y'] = fact['coordinates']['y']
				except Exception as e:
					current['Coordinates.x'] = None
					current['Coordinates.y'] = None
				if event_id == 'GOAL' :
					current['Strength'] = fact['result']['strength']['name']			
				try:
					current['ShotType'] = fact['result']['secondaryType']
				except:
					current['ShotType'] = None
				
				if fact['team']['id'] == home_team:
						current['homeAway'] = 'home'
				else:
						current['homeAway'] = 'away'
				
				
				if current['Period'] < 5: # no shootout shots
					try:
						if fact['team']['id'] == home_team:
							current['rinkSide'] = match['liveData']['linescore']['periods'][fact['about']['period']-1]['home']['rinkSide']
						
						else:
							current['rinkSide'] = match['liveData']['linescore']['periods'][fact['about']['period']-1]['away']['rinkSide']
					except:
					    current['rinkSide'] = None
				else:
					try:
						if current['Coordinates.x'] > 0: # shootout shots
							current['rinkSide'] = 'left'
						else:
							current['rinkSide'] = 'right'
					except:
						current['rinkSide'] = None
				extracted_data.append(current)
	return extracted_data 

def tidy_data(extracted_data) :
	"""
	Format the data with Pandas 
	Returns a Pandas Dataframe

	extracted_data : data extracted returned by the 'extract_data_from_json' function
	"""
	data_frame = pd.json_normalize(extracted_data)
	data_frame["GameDuration"] = pd.to_timedelta(data_frame["GameDuration"])
	return data_frame

# will be userfull later to use our data foreign keys with public endpoints
def get_from_url(url) :
	try :
		return pd.read_json(url)
	except json.decoder.JSONDecodeError :
		print('Import Failed : ' + url)

def tidy_boxscore(boxscore_data):
    """
    Tidies the teams object returned from the boxscore endpoint
    """
    
    def tidy_team(json_data):
        out_dict = {
            "TeamID":json_data["team"]["id"], 
            "TeamName":json_data["team"]["name"],
            "Goals":json_data["teamStats"]["teamSkaterStats"]["goals"],
            "ShotsOnGoal":json_data["teamStats"]["teamSkaterStats"]["shots"]
        }

        return out_dict
    
    away_team = boxscore_data["teams"]["away"]
    home_team = boxscore_data["teams"]["home"]
    return {"Home":tidy_team(home_team), "Away":tidy_team(away_team)}
    
    
def get_boxscore(game_id):
    json_data = fetch_boxscore_data(game_id)
    out_data = tidy_boxscore(json_data)
    out_df = pd.DataFrame.from_dict(out_data)
    
    return out_df

def get_playbyplay_data(json) :
	"""
	Extract usefull data for a given json file
	Returns a list which contains a dictionary for each match with only the usefull data

	json : json string returned by the 'data.import_dataset' function
	"""
	data = json
	extracted_data = []
	for match in data :       
		game_duration = timedelta(minutes=60)
		if len(match['liveData']['linescore']['periods']) > 3: #there is overtime
			game_duration += timedelta(minutes=5)

		home_team_id = int(match['gameData']['teams']['home']['id'])

		for i,fact in enumerate(match['liveData']['plays']['allPlays']):
			current = {}
			event_id = fact['result']['eventTypeId']
			if  event_id == 'SHOT' or event_id == 'GOAL' : # only goals and shots
				current['GamePk'] = match['gamePk']
				current['GameDuration'] = game_duration
				current['EventTypeId'] = event_id
				current['DateTime'] = fact['about']['dateTime']
				current['PeriodTimeRemaining'] = fact['about']['periodTimeRemaining']
				current['Period'] = fact['about']['period']
				current['Team.id'] = fact['team']['id']
				current['Team.name'] = fact['team']['name']
				current['EmptyNet'] = False
				for player in fact['players'] :
					current[player['playerType']] = player['player']['id']
				try:
					current['Coordinates.x'] = fact['coordinates']['x']
					current['Coordinates.y'] = fact['coordinates']['y']
				except Exception as e:
					current['Coordinates.x'] = 0
					current['Coordinates.y'] = 0
				if event_id == 'GOAL' :
					current['Strength'] = fact['result']['strength']['name']			
				try:
					current['ShotType'] = fact['result']['secondaryType']
				except:
					current['ShotType'] = None
				
				if fact['team']['id'] == home_team_id:
						current['homeAway'] = 'home'
				else:
						current['homeAway'] = 'away'

				if event_id == 'GOAL':
					try:
						current['EmptyNet'] = fact['result']['emptyNet']
					except Exception as e:
						continue
				
				
				if current['Period'] < 5: # no shootout shots
					try:
						if fact['team']['id'] == home_team_id:
							current['rinkSide'] = match['liveData']['linescore']['periods'][fact['about']['period']-1]['home']['rinkSide']
						
						else:
							current['rinkSide'] = match['liveData']['linescore']['periods'][fact['about']['period']-1]['away']['rinkSide']
					except:
					    current['rinkSide'] = None
				else:
					try:
						if current['Coordinates.x'] > 0: # shootout shots
							current['rinkSide'] = 'left'
						else:
							current['rinkSide'] = 'right'
					except:
						current['rinkSide'] = None
				
				if current['rinkSide'] == None:
					if current['Coordinates.x'] > 0:
						current['rinkSide'] = 'left'
					else:
						current['rinkSide'] = 'right'
				
				extracted_data.append(current)
	return extracted_data 

def tidy_playbyplay_data(json):

	df = pd.DataFrame(json)

	right_cond = df['rinkSide'] == 'right'
	left_cond = df['rinkSide'] == 'left'

	df_right_side = df[right_cond].copy()
	df_right_side['Coordinates.y'] = np.abs(df_right_side['Coordinates.y'] )

	df_left_side = df[left_cond].copy()
	df_left_side['Coordinates.y'] = np.abs(df_left_side['Coordinates.y'] )

	df_right_side['angle'] = np.rad2deg(np.arctan(df_right_side['Coordinates.y'] / (df['Coordinates.x'] + 89)))
	df_left_side['angle'] = np.rad2deg(np.arctan(df_left_side['Coordinates.y'] / -(df['Coordinates.x'] - 89)))

	df_right_side['goalDist'] = np.linalg.norm(df_right_side[['Coordinates.x', 'Coordinates.y']].to_numpy() + [89,0], 2, axis=1)
	df_left_side['goalDist'] = np.linalg.norm(df_left_side[['Coordinates.x', 'Coordinates.y']].to_numpy() - [89, 0], 2, axis=1)


	for new_col in ['angle', 'goalDist']:
		df.loc[right_cond, new_col] = df_right_side[new_col]
		df.loc[left_cond, new_col] = df_left_side[new_col]

	df['isGoal'] = (df['EventTypeId'] == 'GOAL') * 1
	df['EmptyNet'] = (df['EmptyNet']) * 1
	df.loc[df['Coordinates.y'] < 0, 'angle'] *= -1

	return df[['Coordinates.x', 'Coordinates.y', 'angle', 'goalDist', 'isGoal', 'EmptyNet']]

def tidy_allevents(json):
        
    data = json
    extracted_data = []
    for match in data :       
            previous = {}
            previous['GamePk'] = None
            previous['Period'] = None
            
            start_pp_home = 0
            start_pp_away = 0
            away_pp = []
            home_pp = []

            penalty = 0
            
            home_team_id = int(match['gameData']['teams']['home']['id'])
            away_team_id = int(match['gameData']['teams']['away']['id'])
            
            for i,fact in enumerate(match['liveData']['plays']['allPlays']):
                
                current = {}
                event_id = fact['result']['eventTypeId']
                current['GamePk'] = match['gamePk']
                current['EventTypeId'] = event_id
                current['DateTime'] = fact['about']['dateTime']
                current['Period'] = fact['about']['period']
                
                PeriodTime = fact['about']['periodTime']
                m, s = PeriodTime.split(':')               
                current['TotalSeconds'] = int(int(current['Period'])-1)*1200 + int(m) * 60 + int(s)
                
                try:
                    current['Coordinates.x'] = fact['coordinates']['x']
                    current['Coordinates.y'] = fact['coordinates']['y']
                except Exception as e:
                    current['Coordinates.x'] = 0
                    current['Coordinates.y'] = 0
                
                try:
                    current['team'] = fact['team']['id']
                except Exception as e:
                    current['team'] = ""
                
                # BONUS PART
                
                h_timer = 0
                a_timer = 0
                
                for hpp in home_pp:
                    if len(home_pp) > 0:
                        h_timer = hpp[1] - (current['TotalSeconds']-hpp[2])
                        current['h_timer']=hpp[1]
                        
                for hpp in home_pp:
                    if len(home_pp) > 0:
                        if h_timer <= 0:
                            home_pp.remove(hpp)
                
                if len(home_pp) == 0:
                    start_pp_home = current['TotalSeconds']
                
                
                for app in away_pp:
                    
                    if len(away_pp) > 0:
                        a_timer = app[1] - (current['TotalSeconds']-app[2])
                        current['a_timer']=app[1]
                        
                for app in away_pp:
                    if len(away_pp) > 0:
                        if a_timer <= 0:
                            away_pp.remove(app)
                
                if len(away_pp) == 0:
                    start_pp_away = current['TotalSeconds']


                n_home_players = 5 - len(away_pp.copy())
                n_away_players = 5 - len(home_pp.copy())
                
                try:
                    if fact['team']['id'] == home_team_id:
                        
                        current['TimeSincePP'] = current['TotalSeconds'] - start_pp_home
                        
                        if (event_id == 'GOAL'):
                            for hpp in home_pp:
                                if (hpp[1]>0) and (hpp[0]=='minor'):
                                    home_pp.remove(hpp)
                                    break
                                if (hpp[1]>120) and (hpp[0]=='db_minor'):
                                    hpp[1]=hpp[1]-120
                                    break
                                if (hpp[1]<=120) and (hpp[0]=='db_minor'):
                                    home_pp.remove(hpp)
                                    break   
                        current['FriendPlayers'] = n_home_players
                        current['OpposingPlayers'] = n_away_players                       

                    else:
                        
                        current['TimeSincePP'] = current['TotalSeconds'] - start_pp_away
                        
                        if (event_id == 'GOAL'):
                            for app in away_pp:
                                if (app[1]>0) and (app[0]=='minor'):
                                    away_pp.remove(app)
                                    break
                                if (app[1]>120) and (app[0]=='db_minor'):
                                    app[1]=app[1]-120
                                    break
                                if (app[1]<=120) and (app[0]=='db_minor'):
                                    away_pp.remove(app)
                                    break                 
                        current['FriendPlayers'] = n_away_players
                        current['OpposingPlayers'] = n_home_players

                except Exception as e:
                    current['TimeSincePP'] = 0
                    current['FriendPlayers'] = 5
                    current['OpposingPlayers'] = 5
                

                if current['EventTypeId'] == 'PENALTY':
                    
                    if fact['result']['penaltyMinutes'] == 2:
                        penalty = 120
                        type = 'minor'
                    if fact['result']['penaltyMinutes'] == 4:
                        penalty = 240
                        type = 'db_minor'
                    if fact['result']['penaltyMinutes'] == 5:
                        penalty = 300    
                        type = 'major'
                        
                    if fact['team']['id'] == home_team_id:
                        away_pp.append([type,penalty,current['TotalSeconds']])
                        
                    else:  
                        home_pp.append([type,penalty,current['TotalSeconds']])
                        
 
                # PREVIOUS PLAY

                
                if (current['GamePk'] == previous['GamePk']) and (current['Period'] == previous['Period']):
                        current['Last_event_type'] = previous['EventTypeId']
                        current['Last_coordinates.x'] = previous['Coordinates.x']
                        current['Last_coordinates.y'] = previous['Coordinates.y']
                        current['Time_last_event'] = int(current['TotalSeconds']-previous['TotalSeconds'])
                        current['Distance_last_event'] = ( (current['Coordinates.x']-current['Last_coordinates.x'])**2 +                                                          
                                                         (current['Coordinates.y']-current['Last_coordinates.y'])**2 )**0.5
                        
                        if (current['Last_event_type'] == 'SHOT') and (current['team']==previous['team']):
                            current['Rebound'] = True
                        else:
                            current['Rebound'] = False
                        try:
                            current['Speed'] = round(current['Distance_last_event']/current['Time_last_event'],2)
                        except Exception as e:
                            current['Speed'] = 0
                else:        
                        current['Last_event_type'] = None
                        current['Last_coordinates.x'] = 0
                        current['Last_coordinates.y'] = 0
                        current['Time_last_event'] = 0
                        current['Distance_last_event'] = 0
                        current['Rebound'] = False
                        current['Speed'] = 0       

                
                
                
                extracted_data.append(current)                           
                previous = current.copy() 
                
                
    data = pd.DataFrame(extracted_data)
    data = data[['GamePk','EventTypeId','DateTime','TotalSeconds','Last_event_type','Last_coordinates.x',
                                      'Last_coordinates.y','Time_last_event','Distance_last_event','Rebound','Speed', 'TimeSincePP', 'FriendPlayers', 'OpposingPlayers' ]]
    
    return data

def tidy2_playbyplay_data(json, df):

        new_df = df[['GamePk', 'DateTime', 'EventTypeId', 'Period', 'Coordinates.x', 'Coordinates.y', 'goalDist', 'angle', 'ShotType', 'isGoal' ,'EmptyNet']].copy() 
        new_df = new_df.loc[:,~new_df.columns.duplicated()]
        df_all = tidy_allevents(json)
              
        new_df = pd.merge(new_df,df_all,on=['GamePk','EventTypeId','DateTime'],how='left')
        
        new_df['Change_angle'] = pd.DataFrame(np.zeros((new_df.shape[0],)))
        
        
        for i in range(new_df.shape[0]):
            if new_df.iloc[i,new_df.columns.get_loc('Rebound')]==True:
                new_df.iloc[i,new_df.columns.get_loc('Change_angle')] = round(new_df.iloc[i,new_df.columns.get_loc('angle')] - new_df.iloc[i-1,new_df.columns.get_loc('angle')],2) 
        
        new_df = new_df.drop(columns=['EventTypeId','DateTime'])
        
        
        return new_df