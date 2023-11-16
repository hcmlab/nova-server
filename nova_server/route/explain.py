from pathlib import Path
from flask import Blueprint, request, jsonify
from nova_server.utils.thread_utils import THREADS
from nova_server.utils.job_utils import get_job_id_from_request_form
from nova_server.utils import thread_utils, job_utils
from nova_server.utils import log_utils
from nova_server.utils import env
from nova_server.exec.execution_handler import NovaExplainHandler
import os
import tensorflow as tf
import io
import ast
import numpy as np
import pandas as pd
import io as inputoutput
import base64
import json
import pickle
import warnings
import site as s
from PIL import Image
from PIL import Image as pilimage
from flask import Blueprint, Response, request
from lime.lime_image import LimeImageExplainer
from lime.lime_tabular import LimeTabularExplainer
import tf_explain
import dice_ml

explain = Blueprint("explain", __name__)

@explain.route("/explain", methods=["POST"])
def explain_thread():
        if request.method == "POST":
            request_form = request.form.to_dict()
            job_id = get_job_id_from_request_form(request_form)
            job_added = job_utils.add_new_job(job_id, request_form=request_form)
            thread = explain_data(request_form, job_id)
            thread.start()
            THREADS[job_id] = thread
            data = {"success": str(job_added)}
            return jsonify(data)


@thread_utils.ml_thread_wrapper
def explain_data(request_form, job_id):
    #job_id = get_job_id_from_request_form(request_form)
    job_utils.update_status(job_id, job_utils.JobStatus.RUNNING)
    logger = log_utils.get_logger_for_job(job_id)
    logger.info(request_form)

    job = job_utils.get_job(job_id)
    job.execution_handler = NovaExplainHandler(request_form, logger=logger, backend=os.getenv(env.NOVA_SERVER_BACKEND))

    try:
        job.run()
        job_utils.update_status(job_id, job_utils.JobStatus.FINISHED)
    except Exception as e:
        logger.critical(f"Job failed with exception {str(e)}")
        job_utils.update_status(job_id, job_utils.JobStatus.ERROR)
