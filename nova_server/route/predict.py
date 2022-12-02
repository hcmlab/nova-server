import importlib.util
from nova_server.utils.thread_utils import THREADS
from nova_server.utils.status_utils import update_progress
from nova_server.utils.key_utils import get_key_from_request_form
from nova_server.utils import status_utils, thread_utils, log_utils, dataset_utils, db_utils
from nova_server.utils import polygon_utils
import numpy as np
from pathlib import Path
import nova_server.utils.path_config as cfg
from flask import Blueprint, request, jsonify
from nova_server.utils import thread_utils, status_utils, log_utils, dataset_utils
from nova_server.utils.ssi_utils import Trainer


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
    status_utils.update_status(key, status_utils.JobStatus.RUNNING)
    trainer_file_path = Path(cfg.cml_dir + request_form["trainerFilePath"])

    logger = log_utils.get_logger_for_thread(key)
    logger.info("Action 'Predict' started.")

    trainer = Trainer()

    if not trainer_file_path.is_file():
        logger.error("Trainer file not available!")
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return None
    else:
        trainer.load_from_file(trainer_file_path)
        logger.info("Trainer successfully loaded.")

    if not trainer.model_script_path:
        logger.error('Trainer has no attribute "script" in model tag.')
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return None
    # Load Trainer
    model_script_path = trainer_file_path.parent / trainer.model_script_path
    spec = importlib.util.spec_from_file_location("model_script", model_script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    # Load Data
    try:
        update_progress(key, 'Data loading')
        ds_iter = dataset_utils.dataset_from_request_form(request_form)
        logger.info("Prediction data successfully loaded.")
    except ValueError:
        log_utils.remove_log_from_dict(key)
        logger.error("Not able to load the data from the database!")
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return

    # ToDo scheme type is not necessary. we can use the label_info from the data iterator
    # Load model
    model_weight_path = trainer_file_path.parent / trainer.model_weights_path
    if request_form["schemeType"] == "DISCRETE_POLYGON" or request_form["schemeType"] == "POLYGON":
        data = model_script.preprocess(ds_iter, logger=logger)
        labels = data[1]
        data_list = data[0]
        amount_of_labels = len(labels) + 1
        output_shape = np.uint8(data_list[0][list(data_list[0])[1]])[0].shape
        #model = model_script.load(Path(cfg.cml_dir + "\\" + trainer.model_weights_path), amount_of_labels)
        model = model_script.load(model_weight_path, amount_of_labels)
        # 1. Predict
        confidences_layer = model_script.predict(model, data_list, logger, output_shape)
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
        # 1. Load model
        logger.info("Loading model...")
        model = model_script.load(model_weight_path, logger=logger)
        logger.info("...done")

        # 2. Preprocess data
        logger.info("Preprocessing data...")
        ds_iter_pp = model_script.preprocess(ds_iter, logger=logger, request_form=request_form)
        logger.info("...done")

        # 3. Predict data
        logger.info("Predicting results...")
        results = model_script.predict(model, ds_iter_pp, logger=logger)
        logger.info("...done")

        # 4. Write to database
        logger.info("Uploading to database...")
        db_utils.write_discrete_to_db(request_form, results)
        logger.info("...done")

    elif request_form["schemeType"] == "FREE":
        # 1. Load model
        logger.info("Loading model...")
        model = model_script.load(trainer.model_weights_path, logger=logger)
        logger.info("...done")

        # 2. Preprocess data
        logger.info("Preprocessing data...")
        ds_iter_pp = model_script.preprocess(ds_iter, logger=logger, request_form=request_form)
        logger.info("...done")

        # 3. Predict data
        logger.info("Predicting results...")
        results = model_script.predict(model, ds_iter_pp, logger=logger, request_form=request_form)
        logger.info("...done")

        # 4. Write to database
        logger.info("Uploading to database...")
        db_utils.write_freeform_to_db(request_form, results)
        logger.info("...done")

    elif request_form["schemeType"] == "CONTINUOUS":
        # TODO Marco
        ...
    elif request_form["schemeType"] == "POINT":
        # TODO
        ...

    # TODO tmp files loeschen!!!

    status_utils.update_status(key, status_utils.JobStatus.FINISHED)
