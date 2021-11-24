from comet_ml import API

import pickle
import os 

api = API(api_key=os.getenv('COMET_ML_KEY'))

class ModelLoader() :
    """Class able to load registred models from comet and then open them from specific local folder"""
    class Model_Enum :
        MODEL = 0
        NAME = 1
    def __init__(self, folder_name='Final_Models', username="charlescol") :
        self.folder_name = folder_name
        self.username = username

    def get_models_from_api(self, version='1.0.0') -> None :
        downloaded_files = os.listdir('.\\' + self.folder_name)
        registry_model_names = api.get_registry_model_names(self.username)
        for name in registry_model_names :
            filename = api.get_registry_model_details(self.username, name, version=version)["assets"][0]['fileName']
            if filename not in downloaded_files :
                api.download_registry_model(self.username, name, version, '.\\' + self.folder_name)

    def get_models_from_file(self, extension:str = '.pkl') -> object :
        models = []
        for model_filename in os.listdir('.\\' + self.folder_name) :
            model_pkl = open(f'.\\{self.folder_name}\\{model_filename}', 'rb')
            models.append((pickle.load(model_pkl), model_filename[:-len(extension)]))
        return models