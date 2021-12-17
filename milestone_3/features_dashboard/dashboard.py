from IPython.display import display, clear_output
import sys
import ipywidgets as widgets
from ipywidgets import Box
from ipywidgets import GridspecLayout
from ipywidgets import AppLayout, Button, Layout
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
from milestone_3.features_dashboard import modelLoader, game, process_data#, game_client
from milestone_3.ift6758.ift6758.client import game_client



class dashboard():
    
    
    def __init__(self):
        self.Model_download = modelLoader.ModelLoader()
        self.game = game.Game(0)
        
    def get_game(self,game_id):
        
        if self.game != game_id:
            columns = process_data.get_columns()
            self.complete_df = pd.DataFrame(columns = columns)
            self.game = game.Game(game_id)
            self.restore_api()
            
    
    def define_model(self,model,model_name):
        self.model = model
        self.model_name = model_name
        

    def update_goals(self,df, dmatrix):
        
        index_home = df.index[df['team'] == self.game.home].tolist()
        index_away = df.index[df['team'] == self.game.away].tolist()
        
        df_pred = df.loc[:, df.columns != 'team'] 
        
        if (self.model_name == 'xgb-feature-selection') or (self.model_name == 'xgb-grid-search'):
            pred = self.model.predict(dmatrix)
            pred = np.round(np.array(pred)).astype(int)
        else:
            pred = self.model.predict(df_pred)
                
        
        self.game.home_goal += pred[index_home].sum()
        self.game.away_goal += pred[index_away].sum()
        
        pred = pd.DataFrame(pred)
        pred.columns = ['isGoal']
        new_df = pd.concat((df_pred,pred),axis=1)
        self.complete_df = pd.concat((self.complete_df,new_df), axis=0)
        self.complete_df['isGoal'] =  self.complete_df['isGoal'].astype(int)

    def run_api(self, game_id :int, event_idx:int):

        unseen_events, new_idx = game_client.ping_game(game_id, self.game.last_event) 
                
        return unseen_events, new_idx#, past_events
    
    def restore_api(self):
        game_client.GAME_EVENTS = None
        game_client.GAME_DATA = None
        game_client.GAME_EVENTS = None
        game_client.LAST_IDX = 0
        game_client.GAME_ID = None
    
    
    def update_data(self,game_id):
        
        self.get_game(game_id)
        unseen_events, new_idx = self.run_api(game_id, self.game.last_event)
        if isinstance(unseen_events, pd.DataFrame):
            self.game.last_event = new_idx
            unseen_events, unseen_events_xgb = process_data.process_dataframe(unseen_events)
            self.game.last_event = new_idx
            self.game.update_period_time()
            self.update_goals(unseen_events, unseen_events_xgb)
        else:
            self.team_HOME.description = ''
            self.team_AWAY.description = ''
            self.time.description = 'Invalid game Id or game in state Preview'
            self.GOALS_HOME.description = ''
            self.GOALS_AWAY.description = '' 
    
    def create_layaout(self):
        
        def create_expanded_button(description, button_style):
            return Button(description=description, button_style=button_style, layout=Layout(height='auto', width='auto'))

        # Box workspace name input
        box_workspace = widgets.Dropdown(
            options={'charlescol': 'charlescol'},
            description='Workspace:',
        )

        # Box model name input
        box_model = widgets.Dropdown(
            options={'XGB grid search':'xgb-grid-search','Random Forest': 'q6-random-forest-final', 
                     'SVM': 'q6-svm-final', 'Decision Tree':'q6-decision-tree-final','Naive Bayes': 'q6-naive-bayes-final'},
            description='Model:',
        )


        box_version = widgets.Dropdown(
            options={'1.0.0': '1.0.0'},
            description='Version:',
        )


        # Box game Id input
        box_game_ID = widgets.Text(
            value='',
            description='Game ID:',
            disabled=False
        )


        self.time = create_expanded_button("TIME", 'primary')
        self.team_HOME = create_expanded_button("HOME", 'primary')
        self.team_AWAY = create_expanded_button("AWAY", 'primary')
        self.GOALS_HOME = create_expanded_button("GOALS HOME", 'info')
        self.GOALS_AWAY = create_expanded_button("GOALS AWAY", 'info')

        
        # Button download
        button_download = create_expanded_button("DOWNLOAD/SET Model", 'success')
        log_download = create_expanded_button("Actual model", 'info')
        def on_button_download_clicked(b):
            model = self.Model_download.get_model_from_api(name=box_model.value)
            self.define_model(model,box_model.value )
            log_download.description = self.model_name
        button_download_action = button_download.on_click(on_button_download_clicked)   
        

        # Button ping game
        button_ping = create_expanded_button("PING GAME", 'success')
        log_df = widgets.Output(layout={'border': '1px solid black'})

        def button_ping_clicked(b):            
            self.update_data(int(box_game_ID.value))            
            self.team_HOME.description = self.game.home_name
            self.team_AWAY.description = self.game.away_name
            self.time.description = "Period " + str(self.game.period) +" - " + str(self.game.time_reminder)
            self.GOALS_HOME.description = str(self.game.home_goal)
            self.GOALS_AWAY.description = str(self.game.away_goal)         
            with log_df:
                clear_output()
                display(self.complete_df)
            
            
        button_ping_action = button_ping.on_click(button_ping_clicked)
        
        grid = GridspecLayout(4, 4, justify_content='center')
        grid[0, 0:4] = button_download
        grid[1, 0] = box_workspace
        grid[1, 1] = box_model
        grid[1, 2] = box_version
        grid[2, 0:4] = log_download #output_1_exp
        grid[3, 0] = box_game_ID
        grid[3, 1:4] = button_ping


        grid2 = GridspecLayout(12, 2, justify_content='center')
        grid2[0:3, 0:2] = self.time
        grid2[3:7, 0:1] = self.team_HOME
        grid2[3:7, 1:2] = self.team_AWAY
        grid2[7:11, 0:1] = self.GOALS_HOME
        grid2[7:11, 1:2] = self.GOALS_AWAY

        
        grid3 = GridspecLayout(16, 2, justify_content='center')
        grid3[0:16, 0:4] = log_df

        box_layout = Layout(flex_flow='column',
                            border='solid',
                            justify_content='center')


        items = [grid, grid2, grid3]
        layout = Box(children=items, layout=box_layout)
        return layout