import pathlib
import sys
import importlib
import threading

import numpy as np
from flask import Blueprint, request, jsonify
from nova_server.utils import tfds_utils, thread_utils, status_utils, log_utils
import imblearn


complete = Blueprint("complete", __name__)
thread = Blueprint("thread", __name__)


@complete.route("/complete", methods=["POST"])
def complete_thread():
    if request.method == "POST":
        thread = complete(request.form)
        thread_id = thread.name
        status_utils.add_new_job(thread_id)
        data = {"job_id": thread_id}
        thread.start()
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def complete(request_form):
    trainer_file = request_form.get("trainerScript")

    if trainer_file is None:
        print("TRAINER FILE IS NONE")
        return

    spec = importlib.util.spec_from_file_location("trainer", trainer_file)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    model = None
    model_path = pathlib.Path(request_form.get("trainerPath"))

    try:
        ds_iter = tfds_utils.dataset_from_request_form(request_form)
    except ValueError as ve:
        print("NOT ABLE TO LOAD THE DATA FROM DATABASE")
        return

