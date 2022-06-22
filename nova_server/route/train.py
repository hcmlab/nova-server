import importlib
import threading

import numpy as np
from flask import Blueprint, request, jsonify
from nova_server.utils import tfds_utils, thread_utils, status_utils, log_utils
import imblearn


train = Blueprint("train", __name__)
thread = Blueprint("thread", __name__)


@train.route("/train", methods=["POST"])
def train_thread():
    if request.method == "POST":
        thread = train_model(request.form)
        thread_id = thread.name
        status_utils.add_new_job(thread_id, train.name)
        data = {"job_id": thread_id}
        thread.start()
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def train_model(request_form):

    def update_progress(msg):
        status_utils.update_progress(threading.current_thread().name, msg)

    # Init logging
    status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.RUNNING)

    update_progress('Initalizing')
    logger = log_utils.get_logger_for_thread(__name__)
    log_conform_request = dict(request_form)
    log_conform_request['password'] = '---'
    logger.info(f"Start Training with request {log_conform_request}")

    trainer_file = request_form.get("trainerScript")
    if trainer_file is None:
        logger.error('No trainer file has been provided. Exiting.')
        status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.ERROR)
        return

    logger.info(f"Loading trainer script {trainer_file}...")
    spec = importlib.util.spec_from_file_location("trainer", trainer_file)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    logger.info("... done")

    # Building the dataset
    logger.info("Building dataset from request form...")
    update_progress('Dataloading')

    try:
        ds_iter = tfds_utils.dataset_from_request_form(request_form)
    except ValueError as ve:
        status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.ERROR)
        logger.error('Error when loading dataset {}'.format(str(ve)))
        return
    logger.info("... done")

    # Preprocess data
    logger.info("Start preprocessing...")
    data_list = list(ds_iter)
    data_list.sort(key=lambda x: int(x["frame"].split("_")[0]))
    x = [v[request_form.get("stream").split(" ")[0]] for v in data_list]
    y = [v[request_form.get("scheme").split(";")[0]] for v in data_list]

    x_np = np.ma.concatenate(x, axis=0)
    y_np = np.array(y)
    logger.info("...done")


    if request_form.get("balance") == "over":
        logger.info("Apply oversampling...")
        print("OVERSAMPLING from {} Samples".format(x_np.shape))
        oversample = imblearn.over_sampling.SMOTE()
        x_np, y_np = oversample.fit_resample(x_np, y_np)
        print("to {} Samples".format(x_np.shape))
        logger.info("...done")

    if request_form.get("balance") == "under":
        logger.info("Apply undersampling...")

        print("UNDERSAMPLING from {} Samples".format(x_np.shape))
        undersample = imblearn.under_sampling.RandomUnderSampler()
        x_np, y_np = undersample.fit_resample(x_np, y_np)
        print("to {} Samples".format(x_np.shape))
        logger.info("...done")


    # Load model
    logger.info("Load model...")
    model_path = request_form.get("trainerPath")
    logger.info("...done")

    # Train Model
    update_progress('Training')
    logger.info("Start training...")
    model = trainer.train(x_np, y_np)
    logger.info("...done")

    # Save Model
    update_progress('Saving')
    logger.info("Save model...")
    try:
        trainer.save(model, model_path)
    except AttributeError as err:
        status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.ERROR)
        logger.error('Error when loading dataset {}'.format(str(err)))
        raise err
    logger.info("...done")

    update_progress('Done')
    status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.FINISHED)
