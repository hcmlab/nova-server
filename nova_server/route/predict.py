import importlib.util
import numpy as np
from flask import Blueprint, request, jsonify
from nova_server.utils.thread_utils import THREADS
from nova_server.utils.status_utils import update_progress
from nova_server.utils.key_utils import get_key_from_request_form
from nova_server.utils import status_utils, thread_utils, log_utils, tfds_utils, db_utils
from nova_server.utils import polygon_utils
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from pathlib import Path
import nova_server.utils.path_config as cfg
from flask import Blueprint, request, jsonify
from nova_server.utils import thread_utils, status_utils, log_utils, tfds_utils

predict = Blueprint("predict", __name__)


@predict.route("/predict", methods=["POST"])
def predict_thread():
    if request.method == "POST":
        request_form = request.form.to_dict()
        key = get_key_from_request_form(request_form)
        thread = predict_data(request_form)
        status_utils.add_new_job(key)
        data = {"success": "true"}
        thread.start()
        THREADS[key] = thread
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def predict_data(request_form):
    key = get_key_from_request_form(request_form)
    logger = log_utils.get_logger_for_thread(key)
    logger.info("Action 'Predict' started.")

    trainer_file = Path(cfg.cml_dir + request_form["trainerScript"])

    if trainer_file is None:
        logger.error("Trainer file not available!")
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return
    else:
        logger.info("Trainer file available...")

    spec = importlib.util.spec_from_file_location("trainer", trainer_file)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    try:
        update_progress(key, 'Dataloading')
        ds_iter = tfds_utils.dataset_from_request_form(request_form, mode="predict")
        logger.info("Prediction-Data successfully loaded...")
    except ValueError:
        log_utils.remove_log_from_dict(key)
        logger.error("Not able to load the data from the database!")
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return

    logger.info("Trying to start predictions...")
    # ToDo scheme type is not necessary. we can use the label_info from the data iterator

    if request_form["schemeType"] == "DISCRETE_POLYGON" or request_form["schemeType"] == "POLYGON":
        data_list = list(ds_iter)
        data_list.sort(key=lambda x: int(x[request_form["scheme"]]['name']))
        amount_of_labels = len(ds_iter.label_info[list(ds_iter.label_info)[0]].labels) + 1
        output_shape = np.uint8(data_list[0][list(data_list[0])[1]])[0].shape  # 1 = width, 0 = height
        model = trainer.load(request_form["weightsPath"], amount_of_labels)
        # 1. Predict
        confidences_layer = trainer.predict(model, data_list, logger, output_shape)
        # 2. Create True/False Bitmaps
        binary_masks = polygon_utils.prediction_to_binary_mask(confidences_layer)
        # 3. Get Polygons
        all_polygons = polygon_utils.mask_to_polygons(binary_masks)
        # 4. Get Confidences
        confidences = polygon_utils.get_confidences_from_predictions(confidences_layer, all_polygons)
        # 5. Write to database
        success = db_utils.write_polygons_to_db(request_form, all_polygons, confidences)
        if not success.acknowledged:
            logger.error("An unknown error occurred while writing the date into the database! Try to redo the process.")

    elif request_form["schemeType"] == "DISCRETE":
        # TODO Marco
        ...
    elif request_form["schemeType"] == "FREE":
        # 1. Predict
        #ToDO request_form["weightsPath"] -> is probably wrong
        model = trainer.load(request_form["weightsPath"])
        ds_iter_pp = trainer.preprocess(ds_iter)

        # 2. For sample in ds_iter:
        results = trainer.predict(model, ds_iter_pp, logger=logger)

        # 3. Write to database
        db_utils.write_freeform_to_db(request_form, results)

    elif request_form["schemeType"] == "CONTINUOUS":
        # TODO Marco
        ...
    elif request_form["schemeType"] == "POINT":
        # TODO
        ...

    status_utils.update_status(key, status_utils.JobStatus.FINISHED)
    print()
