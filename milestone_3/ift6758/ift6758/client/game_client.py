import requests
import re
#from ift6758.features import tidy_data as td
from milestone_3.features_dashboard import tidy_data as td
import pandas as pd
import numpy as np
import copy

GAME_DATA = None
GAME_EVENTS = None
LAST_IDX = 0
GAME_ID = None
GAME_URL = "https://statsapi.web.nhl.com/api/v1/game/ID/feed/live"

def ping_game(game_id:int, event_idx:int, *args):
    global GAME_DATA, GAME_ID, LAST_IDX
    
    if GAME_DATA is None or GAME_ID is None or game_id != GAME_ID:
        GAME_DATA = fetch_live_game_data(game_id)
        GAME_EVENTS = GAME_DATA['liveData']['plays']['allPlays']
        GAME_ID = game_id
        LAST_IDX = 0
    
    GAME_EVENTS = GAME_DATA['liveData']['plays']['allPlays'] #added
    
    events = list(filter(lambda e: e['about']['eventIdx'] == event_idx,GAME_EVENTS))
    out = []
    past_out = []
    new_idx = -1
    game_data_past = copy.deepcopy(GAME_DATA)
    game_data_new = copy.deepcopy(GAME_DATA)
    
    events_start = list(filter(lambda e: e['result']['event'] == 'Period Start',GAME_EVENTS))
    if len(events_start):
        game_start = True
    else:
        game_start = False
    
    if (len(events) > 0) and (game_start==True):
        event = events[0]
        new_idx = event['about']['eventIdx']

        if len(GAME_EVENTS) -1 > new_idx:
            new_idx += 1
        
        gd_past = game_data_past['liveData']['plays']['allPlays'][:new_idx]
        past_out = GAME_EVENTS[:new_idx]

        gd_new = game_data_new['liveData']['plays']['allPlays'][new_idx:]
        out = GAME_EVENTS[new_idx:]

        game_data_new['liveData']['plays']['allPlays'] = gd_new
        game_data_past['liveData']['plays']['allPlays'] = gd_past

        LAST_IDX = gd_new[-1]['about']['eventIdx']

        #past_pbp_data = td.get_playbyplay_data([game_data_past])
        #past_pbp_tidied = td.tidy_playbyplay_data(past_pbp_data)
        #past_pbp_tidied2 = td.tidy2_playbyplay_data([game_data_past], pd.concat([pd.DataFrame(past_pbp_data),past_pbp_tidied], axis=1))
        
        new_pbp_data = td.get_playbyplay_data([game_data_new])
        new_pbp_tidied = td.tidy_playbyplay_data(new_pbp_data)
        new_pbp_tidied2 = td.tidy2_playbyplay_data([game_data_new], pd.concat([pd.DataFrame(new_pbp_data),new_pbp_tidied], axis=1))
    
    else:
        new_pbp_tidied2 = None
        LAST_IDX = None
    
    return new_pbp_tidied2, LAST_IDX#, past_pbp_tidied2


    


def fetch_live_game_data(game_id:str):
    """
    Fetches game data from the live game endpoint given a game id.
    Returns a json object corresponding to a game
    
    game_id: game ID to fetch
    """
    
    #print(GAME_URL)
    req_url = re.sub(r'ID', f'{game_id}', GAME_URL)
    resp = requests.get(req_url)
    return resp.json()

#unseen_events, new_idx, past_events = ping_game(2021020329, 8)
#unseen_events, new_idx = ping_game(2021020329, 8)