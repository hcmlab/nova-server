import pathlib
import imblearn
import numpy as np
import importlib.util

from flask import Blueprint, request, jsonify
from nova_server.utils import tfds_utils, thread_utils, status_utils, log_utils
from nova_server.utils.key_utils import get_key_from_request_form


train = Blueprint("train", __name__)
thread = Blueprint("thread", __name__)


def update_progress(key, msg):
    status_utils.update_progress(key, msg)


@train.route("/train", methods=["POST"])
def train_thread():
    if request.method == "POST":
        thread = train_model(request.form)
        status_utils.add_new_job(get_key_from_request_form(request.form))
        data = {"success": "true"}
        thread.start()
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def train_model(request_form):

    status_key = get_key_from_request_form(request_form)
    status_utils.update_status(status_key,
                               status_utils.JobStatus.RUNNING)
    update_progress(status_key, 'Initalizing')

    trainer_file = request_form.get("trainerScript")
    logger, log_key = log_utils.get_logger_for_thread(get_key_from_request_form(request_form))

    log_conform_request = dict(request_form)
    log_conform_request['password'] = '---'

    if trainer_file is None:
        logger.error("Trainer file not available!")
        status_utils.update_status(status_key, status_utils.JobStatus.ERROR)
        return
    else:
        logger.info("Trainer file available...")

    spec = importlib.util.spec_from_file_location("trainer", trainer_file)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    model = None
    model_path = pathlib.Path(request_form.get("trainerPath"))

    try:
        update_progress(status_key, 'Dataloading')
        ds_iter = tfds_utils.dataset_from_request_form(request_form)
        logger.info("Data successfully loaded...")
    except ValueError:
        log_utils.remove_log_from_dict(log_key)
        logger.error("Not able to load the data from the database!")
        status_utils.update_status(status_key, status_utils.JobStatus.ERROR)
        return

    data_list = list(ds_iter)
    if len(data_list) < 5:
        logger.error("The number of available training data is too low! More than four data must be available.")
        status_utils.update_status(status_key, status_utils.JobStatus.ERROR)
        return

    logger.info("Trying to start training...")
    if request_form.get("schemeType") == "DISCRETE_POLYGON" or request_form.get("schemeType") == "POLYGON":
        data_list.sort(key=lambda x: int(x[request_form.get("scheme")]['name']))
        model = trainer.train(data_list, ds_iter.label_info[list(ds_iter.label_info)[0]].labels, logger)
        logger.info("Trained model available!")
    elif request_form.get("schemeType") == "DISCRETE":
        ...
    elif request_form.get("schemeType") == "FREE":
        ...
    elif request_form.get("schemeType") == "CONTINUOUS":
        ...
    elif request_form.get("schemeType") == "POINT":
        # TODO
        ...

    try:
        update_progress(status_key, 'Saving')
        logger.info("Trying to save the model weighs...")
        trainer.save(model, model_path)
        logger.info("Model saved! Path to weights (on server): " + str(pathlib.Path(str(model_path) + ".pth")))
    except AttributeError:
        logger.error("Not able to save the model weights! Maybe the path is denied: " + str(model_path))
        status_utils.update_status(status_key, status_utils.JobStatus.ERROR)
        return

    # TODO Hier abfragen ob complete und wenn nicht gewicht-datei verschieben

    logger.info("Training done!")
    status_utils.update_status(status_key, status_utils.JobStatus.FINISHED)


@thread_utils.ml_thread_wrapper
def train_model_old(request_form):

    # PREPROCESSING
    ds_iter = []
    data_list = list(ds_iter)
    data_list.sort(key=lambda x: int(x["frame"].split("_")[0]))
    x = [v[request_form.get("stream").split(" ")[0]] for v in data_list]
    y = [v[request_form.get("scheme").split(";")[0]] for v in data_list]

    x_np = np.ma.concatenate(x, axis=0)
    y_np = np.array(y)

    # DATA BALANCING
    if request_form.get("balance") == "over":
        print("OVERSAMPLING from {} Samples".format(x_np.shape))
        oversample = imblearn.over_sampling.SMOTE()
        x_np, y_np = oversample.fit_resample(x_np, y_np)
        print("to {} Samples".format(x_np.shape))

    if request_form.get("balance") == "under":
        print("UNDERSAMPLING from {} Samples".format(x_np.shape))
        undersample = imblearn.under_sampling.RandomUnderSampler()
        x_np, y_np = undersample.fit_resample(x_np, y_np)
        print("to {} Samples".format(x_np.shape))
