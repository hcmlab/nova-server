import os
import importlib.util

#import torch.cuda

import nova_server.utils.path_config as cfg

from pathlib import Path
from nova_server.utils import db_utils
from flask import Blueprint, request, jsonify
from nova_server.utils.ssi_utils import Trainer
from importlib.machinery import SourceFileLoader
from nova_server.utils.thread_utils import THREADS
from nova_server.utils.status_utils import update_progress
from nova_server.utils.key_utils import get_key_from_request_form
from nova_server.utils import thread_utils, status_utils, log_utils, dataset_utils

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

    # try:
    logger.info("Action 'Predict' started.")
    status_utils.update_status(key, status_utils.JobStatus.RUNNING)
    sessions = request_form["sessions"].split(";")
    trainer_file_path = Path(cfg.cml_dir + request_form["trainerFilePath"])
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

    

    model_script = None

    # Load Data
    for session in sessions:
        request_form["sessions"] = session  # overwrite so we handle each session seperatly..
        try:
            update_progress(key, 'Data loading')
            ds_iter = dataset_utils.dataset_from_request_form(request_form)
            logger.info("Prediction data successfully loaded.")
        except ValueError:
            log_utils.remove_log_from_dict(key)
            logger.error("Not able to load the data from the database!")
            status_utils.update_status(key, status_utils.JobStatus.ERROR)
            return

        # Load the model_script only once
        if model_script is None:
            # Load Trainer
            model_script_path = trainer_file_path.parent / trainer.model_script_path
            source = SourceFileLoader("model_script", str(model_script_path)).load_module()
            model_script = source.TrainerClass(ds_iter, logger)
            # Set Options 
            logger.info("Setting options...")
            if not request_form["OptStr"] == '':
                for k, v in dict(option.split("=") for option in request_form["OptStr"].split(";")).items():
                    model_script.OPTIONS[k] = v
                    logger.info('...Option: ' + k + '=' + v)
            logger.info("...done.")

        logger.info("Execute preprocessing.")
        model_script.preprocess()
        logger.info("Preprocessing done.")

        if model_script.model is None:
            # Load Model
            model_weight_path = trainer_file_path.parent / trainer.model_weights_path
            logger.info("Loading model...")
            model_script.load(model_weight_path)
            logger.info("...done")
        else:
            model_script.ds_iter = ds_iter

        logger.info("Execute preprocessing.")
        model_script.predict()
        results = model_script.postprocess()
        logger.info("Uploading to database...")
        db_utils.write_annotation_to_db(request_form, results)

        logger.info("...done")

        # logger.info("Writing data to database...")
        # db_utils.write_polygons_to_db(request_form, all_polygons, confidences, logger, start_frame)
        # logger.info("...done")

        # 5. In CML case, delete temporary files..
        if request_form["deleteFiles"] == "True":
            logger.info('Deleting temporary CML files...')
            out_dir = Path(cfg.cml_dir + request_form["trainerOutputDirectory"])
            trainer_name = request_form["trainerName"]
            os.remove(out_dir / trainer.model_weights_path)
            os.remove(out_dir / trainer.model_script_path)
            for f in model_script.DEPENDENCIES:
                os.remove(trainer_file_path.parent / f)
            trainer_fullname = trainer_name + ".trainer"
            os.remove(out_dir / trainer_fullname)
            logger.info('...done')

        logger.info('Prediction completed!')
        status_utils.update_status(key, status_utils.JobStatus.FINISHED)


   # except Exception as e:
    #logger.error('Error:' + str(e))
     #   status_utils.update_status(key, status_utils.JobStatus.ERROR)
    #finally:
    #    del results, ds_iter, ds_iter_pp, model, model_script, model_script_path, model_weight_path, spec
