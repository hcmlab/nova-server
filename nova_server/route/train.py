import importlib
import threading

import numpy as np
from flask import Blueprint, request, jsonify
from nova_server.utils import tfds_utils, thread_utils, status_utils
import imblearn
import logging


train = Blueprint("train", __name__)
thread = Blueprint("thread", __name__)


@train.route("/train", methods=["POST"])
def train_thread():
    if request.method == "POST":
        id = train_model(request.form)
        status_utils.add_new_job(id)
        data = {"job_id": id}
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def train_model(request_form):

    # Init logging
    logger = logging.getLogger(threading.current_thread().name)

    spec = importlib.util.spec_from_file_location(
        "trainer", request_form.get("trainerScript")
    )
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    # Building the dataset
    ds, ds_info = tfds_utils.dataset_from_request_form(request_form)

    # Preprocess data
    data_it = ds.as_numpy_iterator()
    data_list = list(data_it)
    data_list.sort(key=lambda x: int(x["frame"].decode("utf-8").split("_")[0]))
    x = [v[request_form.get("stream").split(" ")[0]] for v in data_list]
    y = [v[request_form.get("scheme").split(";")[0]] for v in data_list]

    x_np = np.ma.concatenate(x, axis=0)
    y_np = np.array(y)

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

    # Load model
    modelpath = request_form.get("trainerPath")

    # Train Model
    model = trainer.train(x_np, y_np)

    # Save Model
    trainer.save(model, modelpath)
