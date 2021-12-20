from comet_ml import API
import pickle
import os 

from dotenv import load_dotenv
load_dotenv()

api = API(api_key=os.getenv('COMET_ML_KEY'))

class ModelLoader() :
    """Class able to load registred models from comet and then open them from specific local folder"""
    class Model_Enum :
        MODEL = 0
        NAME = 1
    def __init__(self, folder_name='Final_Models', username="charlescol") :
        self.folder_name = folder_name
        self.username = username
        
        self.dic_models = {
        'xgb_model_grid.pkl' : 'xgb_model_grid.pkl',
        'q6-decision-tree-final' : 'Final_DT.pkl',
        'q6-naive-bayes-final' : 'Final_NB.pkl',
        'q6-random-forest-final' : 'Final_RF.pkl',
        'q6-svm-final' : 'Final_SVM.pkl',
        'xgb-feature-selection' : 'xgb_model_feat_sel.pkl'}
   
    def get_model_from_api(self, name, version='1.0.0') -> None :
        downloaded_files = os.listdir()#('.\\' + self.folder_name)
        filename = api.get_registry_model_details(self.username, name, version=version)["assets"][0]['fileName']
        if filename not in downloaded_files :
            api.download_registry_model(self.username, name, version, '.\\' + self.folder_name)
        model_filename = filename
        extension = '.pkl'
    #def get_model(self, model_filename, extension:str = '.pkl') -> object :
        model_pkl = open(f'{self.folder_name}\\{model_filename}', 'rb')       
        model_pkl = pickle.load(model_pkl)#, model_filename[:-len(extension)]
        return model_pkl