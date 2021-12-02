"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from os.path import dirname, abspath
from pathlib import Path
import logging
from typing import Optional

from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
from waitress import serve

from milestone_3.ift6758.ift6758.client.serving_client import ServingClient

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")


app = Flask(__name__)

client_service: Optional[ServingClient] = None


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    file_dir = dirname(abspath(__file__))
    log_file_path = Path(file_dir) / LOG_FILE

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
                        level=logging.INFO,
                        force=True)

    # TODO: any other initialization before the first request (e.g. load default model)
    global client_service
    client_service = ServingClient(log_file_path=log_file_path)


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # TODO: read the log file specified and return the data
    if client_service is None:
        raise RuntimeError('Client Service is not yet instantiated!')

    return jsonify(client_service.logs())  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # TODO: check to see if the model you are querying for is already downloaded

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    
    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    raise NotImplementedError("TODO: implement this endpoint")

    response = None

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # TODO:
    raise NotImplementedError("TODO: implement this enpdoint")
    
    response = None

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!

# logger = logging.getLogger(__name__)
# logger.info('dddd')


if __name__ == '__main__':
    serve(app, listen='*:8080')
