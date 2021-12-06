import os
import pickle
import traceback
from os.path import dirname, abspath
from pathlib import Path
from typing import Union, Any

import pandas as pd
import logging

from comet_ml import API
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, log_file_path: Path, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization
        self.log_file_path = log_file_path
        self.comet_api = API(api_key=os.getenv('COMET_ML_KEY'))
        self.predictor: Any = None

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        if self.predictor is None:
            raise RuntimeError('No model loaded. Call /download_registry_model first!')
        return pd.DataFrame(self.predictor.predict_proba(X), columns=['No Goal', 'Goal'])

    def logs(self) -> dict:
        """Get server logs"""
        return {
            'logs': self.log_file_path.read_text().splitlines()
        }

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        err_data = {}
        models_folder_path = self.__get_models_folder_path()
        try:
            self.comet_api.download_registry_model(workspace, model, version, output_path=str(models_folder_path))
            self.load_predictor(workspace, model, version)
            success = True
            logger.info(f'Loaded model {model} V{version} from workspace: {workspace}.')
        except Exception:
            success = False
            error_msg = 'Failed to download and load model.'
            logger.exception(error_msg)
            err_data['error'] = traceback.format_exc()

        return {
            'success': success,
                **err_data
        }

    def is_model_downloaded(self, workspace: str, model: str, version: str) -> bool:
        return self.__get_model_file_path(workspace, model, version).exists()

    def __get_model_file_path(self, workspace: str, model: str, version: str) -> Path:
        return self.__get_models_folder_path() / self.__get_model_file_name(workspace, model, version)

    @staticmethod
    def __get_models_folder_path() -> Path:
        return Path(dirname(dirname(abspath(__file__)))) / 'data'

    def __get_model_file_name(self, workspace: str, model: str, version: str) -> str:
        registry_details = self.comet_api.get_registry_model_details(
            workspace,
            model,
            version=version
        )
        if registry_details is not None:
            return registry_details["assets"][0]['fileName']
        else:
            raise RuntimeError(f'Model not found in comet ml registry. '
                               f'Workspace {workspace}, model: {model}, version {version}.')

    def load_predictor(self, workspace: str, model: str, version: str):
        self.__load_predictor(self.__get_model_file_path(workspace, model, version))

    def __load_predictor(self, file_path: Union[Path, str]):
        self.predictor = pickle.load(open(file_path, 'rb'))
