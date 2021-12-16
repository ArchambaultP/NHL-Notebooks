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

from milestone_3.features_dashboard import modelLoader, game, test_naive_api, process_data


class dashboard():
    
    
    def __init__(self):
        self.Model_download = modelLoader.ModelLoader()
        self.game = 0
        
    def get_game(self,game_id):
        
        if self.game != game_id:
            columns = process_data.get_columns()
            self.complete_df = pd.DataFrame(columns = columns)
     
        self.game = game.Game(game_id)
    
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
        
        pred = pd.DataFrame(pred).astype(int)
        pred.columns = ['isGoal']
        new_df = pd.concat((df_pred,pred),axis=1)
        self.complete_df = pd.concat((self.complete_df,new_df), axis=0)


    def run_api(self, game_id :int, event_idx:int):

        unseen_events, new_idx, past_events = test_naive_api.ping_game(game_id, event_idx=event_idx) 

        return unseen_events, new_idx, past_events
    
    def update_data(self,game_id):
        
        self.get_game(game_id)
        unseen_events, new_idx, past_events = self.run_api(game_id, 30)
        unseen_events, unseen_events_xgb = process_data.process_dataframe(unseen_events)
        self.game.last_event = new_idx
        #self.game.update_period_time()
        self.update_goals(unseen_events, unseen_events_xgb)


    
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

        # Box version name input
        box_version = widgets.Text(
            value='1',
            description='Version:',
            disabled=False
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

        ##########################################test
        Period = 3
        Time_left = '2:15'
        #time = create_expanded_button("Period " + str(Period) +" - " + Time_left, 'danger')
        time = widgets.Output(layout={'border': '1px solid black'})
        ##########################################test

        box_layout1 = Layout(display='flex',
                            flex_flow='row',
                            align_items='stretch',
                            border='solid',
                            width='100%')

        team_HOME = widgets.Output(layout={'border': '1px solid black'}, style={'background-color': 'red'})
        

        #team_HOME = widgets.Output(style={'background-color': 'red'})
        team_AWAY = widgets.Output(layout={'border': '1px solid black'})
        team1 = widgets.Output(layout={'border': '1px solid black'})
        team2 = widgets.Output(layout={'border': '1px solid black'})
        GOALS_HOME = widgets.Output(layout={'border': '1px solid black'})
        GOALS_AWAY = widgets.Output(layout={'border': '1px solid black'})


        #items = [time]
        #box = Box(children=items, layout=box_layout1)
        
        
        # Button download
        button_download = create_expanded_button("DOWNLOAD", 'success')
        log_download = widgets.Output(layout={'border': '1px solid black'})
        def on_button_download_clicked(b):
            model = self.Model_download.get_model_from_api(name=box_model.value)
            self.define_model(model,box_model.value )
            #self.define_model(get_model(model_filename=box_model.value))
            with log_download:
                clear_output()
                print('actual model: '+str(box_model.value) )
        button_download_action = button_download.on_click(on_button_download_clicked)   
        
        '''
        b=widgets.HTML(
        value="Hello <p>World</p><p>World</p><p>World</p><p>World</p><p>World</p><p>World</p><p>World</p><p>World</p>",
        placeholder='Some HTML',
        description='Some HTML',
        disabled=True)
        log_df = HBox([b], layout=Layout(height='50px', overflow_y='auto'))
        '''
        # Button ping game
        button_ping = create_expanded_button("Ping game", 'success')
        log_df = widgets.Output(layout={'border': '1px solid black'})
        #log_df.layout.overflow = 'visible'
        def button_ping_clicked(b):
            self.update_data(int(box_game_ID.value))
            with team1:
                clear_output()
                print(self.game.home_name)
            with team2:
                clear_output()
                print(self.game.away_name) 
            #with log_ping:
            #    clear_output()
            #    print(str(box_game_ID.value)+' event X')
            with time:
                clear_output()
                print("Period " + str(Period) +" - " + Time_left)
            with GOALS_HOME:
                clear_output()
                print(self.game.home_goal)
            with GOALS_AWAY:
                clear_output()
                print(self.game.away_goal)
            with log_df:
                display(self.complete_df)
        button_ping_action = button_ping.on_click(button_ping_clicked)
        
        grid = GridspecLayout(4, 4)
        grid[0, 0:4] = button_download
        grid[1, 0] = box_workspace
        grid[1, 1] = box_model
        grid[1, 2] = box_version
        grid[2, 0:4] = log_download #output_1_exp
        grid[3, 0] = box_game_ID
        grid[3, 1:4] = button_ping
        #grid[4, 0:4] = log_ping

        grid2 = GridspecLayout(16, 2)
        grid2[0:4, 0:2] = time
        grid2[4:8, 0:1] = team_HOME
        grid2[4:8, 1:2] = team_AWAY
        grid2[8:12, 0:1] = team1
        grid2[8:12, 1:2] = team2
        grid2[12:16, 0:1] = GOALS_HOME
        grid2[12:16, 1:2] = GOALS_AWAY
        
        grid3 = GridspecLayout(16, 2)
        grid3[0:16, 0:4] = log_df
        
        
        '''
        box_layout = widgets.Layout(display='flex',
                            flex_flow='row',
                            border='1px solid black',
                            width='140px',
                            height='400px')
        
        '''
        box_layout = Layout(#display='flex',
                            flex_flow='column',
                            #align_items='stretch',
                            border='solid')


        items = [grid, grid2]
        layout = Box(children=items, layout=box_layout)
        return layout