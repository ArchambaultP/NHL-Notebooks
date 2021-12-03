"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import json
import logging
import os
import traceback
from os.path import dirname, abspath
from pathlib import Path
from typing import Optional

import pandas as pd
from flask import Flask, jsonify, request, abort
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
    params = request.get_json()
    app.logger.info(params)
    workspace, model, version = params['workspace'], params['model'], params['version']

    try:
        is_model_downloaded = client_service.is_model_downloaded(workspace, model, version)
    except RuntimeError:
        app.logger.exception('')
        return abort(404, traceback.format_exc())

    if is_model_downloaded:
        client_service.load_predictor(workspace, model, version)
        app.logger.info('Model is already downloaded. Loaded model from local file.')
        response = {
            'success': True
        }
    else:
        response = client_service.download_registry_model(workspace, model, version)

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    params = request.get_json()
    app.logger.info(params)

    X = pd.read_json(str(params), orient='values')
    try:
        predictions = client_service.predict(X)
    except RuntimeError:
        app.logger.exception('')
        return abort(400, traceback.format_exc())

    response = json.loads(predictions.to_json(orient='columns'))

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


if __name__ == '__main__':
    serve(app, listen='*:8080')
