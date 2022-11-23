import os
import shutil
import pathlib
import imblearn
import numpy as np
import importlib.util
import xml.etree.ElementTree as ET

from flask import Blueprint, request, jsonify
from nova_server.utils import tfds_utils, thread_utils, status_utils, log_utils
from nova_server.utils.status_utils import update_progress
from nova_server.utils.key_utils import get_key_from_request_form
from nova_server.utils.thread_utils import THREADS
from pathlib import Path
import nova_server.utils.path_config as cfg

train = Blueprint("train", __name__)
thread = Blueprint("thread", __name__)


@train.route("/train", methods=["POST"])
def train_thread():
    if request.method == "POST":
        request_form = request.form.to_dict()
        key = get_key_from_request_form(request_form)
        thread = train_thread_function(request_form)
        status_utils.add_new_job(key)
        data = {"success": "true"}
        thread.start()
        THREADS[key] = thread
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def train_thread_function(request_form):
    train_model(request_form)


def train_model(request_form):
    key = get_key_from_request_form(request_form)
    status_utils.update_status(key, status_utils.JobStatus.RUNNING)
    update_progress(key, 'Initializing')

    trainer_file = Path(cfg.cml_dir + request_form["trainerScript"])
    logger = log_utils.get_logger_for_thread(key)

    log_conform_request = dict(request_form)
    log_conform_request['password'] = '---'

    logger.info("Action 'Train' started.")

    if trainer_file is None:
        logger.error("Trainer file not available!")
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return None
    else:
        logger.info("Trainer file available...")

    spec = importlib.util.spec_from_file_location("trainer", trainer_file)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    model = None
    model_path = Path(cfg.cml_dir + request_form["trainerPath"])
    #model_path = pathlib.Path(request_form["trainerPath"])

    try:
        update_progress(key, 'Data loading')
        ds_iter = tfds_utils.dataset_from_request_form(request_form)
        logger.info("Train-Data successfully loaded...")
    except ValueError:
        log_utils.remove_log_from_dict(key)
        logger.error("Not able to load the data from the database!")
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return None

    data_list = list(ds_iter)
    #TODO: WTF?
    #if len(data_list) < 5:
    #    logger.error("The number of available training data is too low! More than four data must be available.")
    #    status_utils.update_status(key, status_utils.JobStatus.ERROR)
    #    return

    logger.info("Trying to start training...")
    if request_form["schemeType"] == "DISCRETE_POLYGON" or request_form["schemeType"] == "POLYGON":
        data_list.sort(key=lambda x: int(x[request_form["scheme"]]['name']))
        model = trainer.train(data_list, ds_iter.label_info[list(ds_iter.label_info)[0]].labels, logger)
        logger.info("Trained model available!")
    elif request_form["schemeType"] == "DISCRETE":
        x_np, y_np = preprocess_data(request_form, data_list)
        # TODO Marco
    elif request_form["schemeType"] == "FREE":
        x_np, y_np = preprocess_data(request_form, data_list)
        # TODO Marco
    elif request_form["schemeType"] == "CONTINUOUS":
        x_np, y_np = preprocess_data(request_form, data_list)
        # TODO Marco
    elif request_form["schemeType"] == "POINT":
        # TODO
        ...

    try:
        update_progress(key, 'Saving')
        logger.info("Trying to save the model weights...")
        trainer.save(model, model_path)
        # TODO: WTF? PTH?
        logger.info("Model saved! Path to weights (on server): " + str(pathlib.Path(str(model_path) + ".pth")))
    except AttributeError:
        logger.error("Not able to save the model weights! Maybe the path is denied: " + str(model_path))
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return

    delete_unnecessary_files(model_path)
    update_trainer_file(Path(cfg.cml_dir + request_form["templatePath"]))
    # TODO: WTF?
    #trainer_file_path = pathlib.Path.joinpath(model_path.parents[0], 'models', 'trainer',
    #                                          request_form["schemeType"].lower(), request_form["scheme"],
    #                                          request_form["streamType"] + "{" + request_form["streamName"] + "}",
    #                                          request_form["trainerScriptName"])
    #trainer_file_path = model_path
    move_files(weights_path=pathlib.Path(str(model_path) + ".pth"), trainer_path=pathlib.Path((cfg.cml_dir + request_form["templatePath"])),
               out_path=model_path.parent, logger=logger)
    logger.info("Training done!")
    if request_form['mode'] == "TRAIN":
        status_utils.update_status(key, status_utils.JobStatus.FINISHED)

    # Returning the weights-path, in case we want in the next step predict with it
    return model_path.parent / (model_path.name + '.pth')


# Returns the
def move_files(weights_path, trainer_path, out_path, logger):

    # ToDo Delete only files!
    #if os.path.exists(out_path.parent):
    #    shutil.rmtree(out_path.parent)
    #out_path.mkdir(parents=True, exist_ok=True)
    #logger.info("Output path (out_path) created.")
    trainer_path = trainer_path.parents[0]
    # Step 1: Move weights file
    #os.replace(weights_path, os.path.join(out_path, os.path.basename(weights_path)))

    # Step 2: Copy trainer files
    for filename in os.listdir(trainer_path):
        file = os.path.join(trainer_path, filename)
        if os.path.isfile(file):
            shutil.copy(file, os.path.join(out_path, filename))


def update_trainer_file(trainer_path):
    root = ET.parse(pathlib.Path(trainer_path))
    info = root.find('info')
    info.set('trained', 'true')


def delete_unnecessary_files(path):
    weights_file_name = os.path.basename(path)
    path_to_files = path.parents[0]
    for filename in os.listdir(path_to_files):
        file = os.path.join(path_to_files, filename)
        if os.path.isfile(file) and filename.split('.')[0] != weights_file_name:
            os.remove(file)


def preprocess_data(request_form, data_list):
    data_list.sort(key=lambda x: int(x["frame"].split("_")[0]))
    x = [v[request_form["streamName"].split(" ")[0]] for v in data_list]
    y = [v[request_form["scheme"].split(";")[0]] for v in data_list]

    return np.ma.concatenate(x, axis=0), np.array(y)


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