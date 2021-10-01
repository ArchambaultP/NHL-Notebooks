import json
import pandas as pd

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
		for fact in match['liveData']['plays']['allPlays'] :
			current = {}
			event_id = fact['result']['eventTypeId']
			if  event_id == 'SHOT' or event_id == 'GOAL' : # only goals and shots
				current['GamePk'] = match['gamePk']
				current['EventTypeId'] = event_id
				current['DateTime'] = fact['about']['dateTime']
				current['PeriodTimeRemaining'] = fact['about']['periodTimeRemaining']
				current['Team.id'] = fact['team']['id']
				for player in fact['players'] :
					current[player['playerType']] = player['player']['id']
				current['Coordinates.x'] = fact['coordinates']['x']
				current['Coordinates.y'] = fact['coordinates']['y']
				if event_id == 'GOAL' :
					current['Strength'] = fact['result']['strength']['name']
				extracted_data.append(current)
	return extracted_data 

def tidy_data(extracted_data) :
	"""
	Format the data with Pandas 
	Returns a Pandas Dataframe

	extracted_data : data extracted returned by the 'extract_data_from_json' function
	"""
	data_frame = pd.json_normalize(extracted_data)
	return data_frame

# will be userfull later to use our data foreign keys with public endpoints
def get_from_url(url) :
	try :
		return pd.read_json(url)
	except json.decoder.JSONDecodeError :
		print('Import Failed : ' + url)


