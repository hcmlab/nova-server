import os
import shutil
import pathlib
import imblearn
import numpy as np
import importlib.util
import xml.etree.ElementTree as ET

from flask import Blueprint, request, jsonify
from nova_server.utils import dataset_utils, thread_utils, status_utils, log_utils
from nova_server.utils.status_utils import update_progress
from nova_server.utils.key_utils import get_key_from_request_form
from nova_server.utils.thread_utils import THREADS
from pathlib import Path
from nova_server.utils.ssi_utils import Trainer

import nova_server.utils.path_config as cfg

train = Blueprint("train", __name__)
thread = Blueprint("thread", __name__)


@train.route("/train", methods=["POST"])
def train_thread():
    if request.method == "POST":
        request_form = request.form.to_dict()
        key = get_key_from_request_form(request_form)
        thread = train_model(request_form)
        status_utils.add_new_job(key)
        data = {"success": "true"}
        thread.start()
        THREADS[key] = thread
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def train_model(request_form):
    key = get_key_from_request_form(request_form)
    status_utils.update_status(key, status_utils.JobStatus.RUNNING)
    update_progress(key, 'Initializing')

    trainer_file_path = Path(cfg.cml_dir + request_form["trainerFilePath"])
    out_dir = Path(cfg.cml_dir + request_form["trainerOutputDirectory"])
    trainer_name = request_form["trainerName"]

    logger = log_utils.get_logger_for_thread(key)
    log_conform_request = dict(request_form)
    log_conform_request['password'] = '---'

    logger.info("Action 'Train' started.")
    trainer = Trainer()

    if not trainer_file_path.is_file():
        logger.error("Trainer file not available!")
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return None
    else:
        trainer.load_from_file(trainer_file_path)
        logger.info("Trainer successfully loaded.")

    # Load Trainer
    model_script_path = trainer_file_path.parent / trainer.model_script_path
    spec = importlib.util.spec_from_file_location("model_script", model_script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    # Load Data
    try:
        update_progress(key, 'Data loading')
        ds_iter = dataset_utils.dataset_from_request_form(request_form)
        logger.info("Train-Data successfully loaded...")
    except ValueError:
        log_utils.remove_log_from_dict(key)
        logger.error("Not able to load the data from the database!")
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return None

    model = None
    try:
        logger.info("Preprocessing data...")
        # TODO: generic preprocessing interface, remove request form from model interface
        data = model_script.preprocess(ds_iter, logger=logger, request_form=request_form)
        logger.info("...done")
    except ValueError:
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return

    logger.info("Training model...")
    model = model_script.train(data, logger)
    logger.info("...done")

    # Save
    logger.info('Saving...')

    out_dir.mkdir(parents=True, exist_ok=True)

    trainer.info_trained = True
    trainer.model_weights_path = trainer_name
    # TODO add classes and users / sessions
    trainer.write_trainer_to_file(out_dir / trainer_name)
    logger.info('...trainerfile')
    model_script.save(model, out_dir / trainer_name)
    logger.info('...weights')
    shutil.copy(model_script_path, out_dir / trainer.model_script_path)
    logger.info('...train script')
    for f in model_script.DEPENDENCIES:
        shutil.copy(trainer_file_path.parent / f, out_dir / f)
    logger.info('...dependencies')
    logger.info("Training completed!")
    status_utils.update_status(key, status_utils.JobStatus.FINISHED)


# Returns the weights path
def copy_files(files_to_move, files_path, out_path):
    out_path.mkdir(parents=True, exist_ok=True)
    new_weights_path = None

    for file in files_to_move:
        old_file_path = os.path.join(files_path, file)
        new_file_path = os.path.join(out_path, file)
        shutil.copy(old_file_path, new_file_path)

    return new_weights_path


def update_trainer(trainer):
    trainer.info_trained = True


    root = ET.parse(pathlib.Path(trainer_path))
    info = root.find('info')
    model = root.find('model')
    info.set('trained', 'true')
    model.set('path', str(weights_path))
    root.write(trainer_path)
    print()


def delete_unnecessary_files(path):
    weights_file_name = os.path.basename(path)
    path_to_files = path.parents[0]
    for filename in os.listdir(path_to_files):
        file = os.path.join(path_to_files, filename)
        if os.path.isfile(file) and filename.split('.')[0] != weights_file_name:
            os.remove(file)


# TODO DATA BALANCING
def balance_data(request_form, x_np, y_np):
    # DATA BALANCING
    if request_form["balance"] == "over":
        print("OVERSAMPLING from {} Samples".format(x_np.shape))
        oversample = imblearn.over_sampling.SMOTE()
        x_np, y_np = oversample.fit_resample(x_np, y_np)
        print("to {} Samples".format(x_np.shape))

    if request_form["balance"] == "under":
        print("UNDERSAMPLING from {} Samples".format(x_np.shape))
        undersample = imblearn.under_sampling.RandomUnderSampler()
        x_np, y_np = undersample.fit_resample(x_np, y_np)
        print("to {} Samples".format(x_np.shape))
