import pandas as pd
from pandas import DataFrame
import plotly as pl
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer
import importlib

from ift6758.client import game_client

class Game():
    
    def __init__(self,game_id):
        if game_id != 0:
            self.game_data = game_client.fetch_live_game_data(game_id)
            self.away = self. game_data['gameData']['teams']['away']['id']
            self.home = self. game_data['gameData']['teams']['home']['id']
            self.away_name = self. game_data['gameData']['teams']['away']['name']
            self.home_name = self. game_data['gameData']['teams']['home']['name']
        self.home_goal = 0
        self.away_goal = 0
        self.period = 1
        self.time_reminder = 20
        self.last_event = 0
        
    def update_period_time(self):        
        events = list(filter(lambda e: e['about']['eventIdx'] == int(self.last_event),self.game_data['liveData']['plays']['allPlays']))
        event = events[0]
        self.period = event['about']['period']
        self.time_reminder = event['about']['periodTimeRemaining']
        