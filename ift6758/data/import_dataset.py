import re
import requests
import json
import os
"""
Game Types:
01 = preseason
02 = regular season
03 = playoffs
04 = all-star

YYYYTPXXXX
"""
GAME_TYPES = {1: "preseason", 2:"regular", 3:"playoffs", 4:"allstar"}

def import_dataset(year:int, game_type:int):
    """
    Fetches all the game data for a given year corresponding to the start of the season (2016 => 2016-2017 season) and a given game type.
    Returns a list of json objects corresponding to games for the whole season
    
    year: start of season
    game_type: type of games to fetch.
    """
    
    if year < 2017: #hard coding the number of games in a regular season
        n_games = 1230
    else:
        n_games = 1271
    
    # create the dataset directory on this path
    os.makedirs("dataset", exist_ok=True)
    file_name = f"dataset/{year}_{GAME_TYPES[game_type]}.json"
    
    if os.path.isfile(file_name): # file exists
        with open(file_name, "r") as f:
            return json.load(f)
    
    json_data = [] # need to store json data within a top container
    
    if game_type in [1,2]: #preseason/regular
        for i in range(1,n_games+1):
            game_id = f"{year}0{game_type}{str(i).zfill(4)}"
            json_data.append(fetch_live_game_data(game_id))
            
    elif game_type == 3: #playoffs
        """
        instead of listing gamePk manually for playoffs, fetch all the game ids on schedule and then query the game api for data
        """
        
        season = f"{year}{year+1}"
        schedule_url = f"https://statsapi.web.nhl.com/api/v1/schedule?season={season}&gameType=P"
        
        resp = requests.get(schedule_url)
        data = resp.json()
        
        game_ids = []
        for date in data["dates"]:
            for game in date["games"]:
                game_ids.append(str(game["gamePk"]))
        
        for gid in game_ids:
            json_data.append(fetch_live_game_data(gid))
        
    else:
        print("No data")
        return []
    
    with open(file_name,"w") as f:
        json.dump(json_data, f)
    return json_data

def fetch_live_game_data(game_id:str):
    """
    Fetches game data from the live game endpoint given a game id.
    Returns a json object corresponding to a game
    
    game_id: game ID to fetch
    """
    
    url = "https://statsapi.web.nhl.com/api/v1/game/ID/feed/live"
    req_url = re.sub(r'ID', game_id, url)
    print(f"fetching {req_url}")
    resp = requests.get(req_url)
    return resp.json()
