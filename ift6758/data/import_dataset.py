import re
from enum import Enum
from typing import List, Dict, Any, Optional

import requests
import json
from pathlib import Path
from os.path import dirname, abspath
from ift6758.ressources import local_ressource as url


class GameType(str, Enum):
    preseason = 'PR'
    regular = 'R'
    playoffs = 'P'
    allstar = 'A'

    def __str__(self):
        return f'{self.value}'


def import_dataset(year: int, game_type: GameType, path=None, returnData=False) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches all the game data for a given year corresponding to the start of the season (2016 => 2016-2017 season) and a given game type.
    The game data is then saved to a given path. If the path is omitted, the data is going to be saved at the module root
    Returns a list of json objects corresponding to games for the whole season when the returnData flag is set to True

    year: start of season
    game_type: type of games to fetch. see enum above.
    path: path to save dataset
    """

    if path is None:
        root = Path(dirname(abspath(__file__)))
    else:
        root = Path(path)

    data_dir = root / "dataset"
    file_name = f"{year}_{game_type.name}.json"
    file_path = data_dir / file_name
    # checking if dataset does not exist at given path.
    if not data_dir.exists():
        data_dir.mkdir()

    # checking files exist
    if file_path.exists():
        with file_path.open("r") as f:
            if returnData:
                return json.load(f)
            else:
                return

    """
    instead of listing gamePk manually for games, fetch all the game ids on schedule and then query the game api for data
    """

    season = f"{year}{year + 1}"

    schedule_url = f"{url.schedule_endpoint}?season={season}&gameType={game_type.value}"
    print(f"fetching {schedule_url}")
    resp = requests.get(schedule_url)
    data = resp.json()

    game_ids = []
    for date in data["dates"]:
        for game in date["games"]:
            game_ids.append(str(game["gamePk"]))

    json_data = []  # need to store json data in a top container
    for gid in game_ids:
        data = fetch_live_game_data(gid)
        json_data.append(data)

    with file_path.open("w") as f:
        json.dump(json_data, f)

    if returnData:
        return json_data


def fetch_live_game_data(game_id: str) -> Dict[str, Any]:
    """
    Fetches game data from the live game endpoint given a game id.
    Returns a json object corresponding to a game

    game_id: game ID to fetch
    """

    req_url = re.sub(r'ID', game_id, url.games_endpoint)
    print(f"Fetching {req_url}")
    resp = requests.get(req_url)
    return resp.json()
