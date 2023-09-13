"""General logic for predicting annotations to the nova database
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 06.09.2023
"""


from pathlib import Path
from flask import Blueprint, request, jsonify
from nova_server.utils.thread_utils import THREADS
from nova_server.utils.job_utils import get_job_id_from_request_form
from nova_server.utils import thread_utils, job_utils
from nova_server.utils import log_utils
from nova_server.exec.execution_handler import NovaPredictHandler

predict = Blueprint("predict_in_venv", __name__)


@predict.route("/predict_in_venv", methods=["POST"])
def predict_thread():
    if request.method == "POST":
        request_form = request.form.to_dict()
        key = get_job_id_from_request_form(request_form)
        job_utils.add_new_job(key, request_form=request_form)
        thread = extract_data(request_form)
        thread.start()
        THREADS[key] = thread
        data = {"success": "true"}
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def extract_data(request_form):
    key = get_job_id_from_request_form(request_form)

    job_utils.update_status(key, job_utils.JobStatus.RUNNING)
    logger = log_utils.get_logger_for_job(key)
    handler = NovaPredictHandler(request_form, logger=logger)

    # TODO replace .env with actual path
    dotenv_path = Path(".env").resolve()

    try:
        handler.run(dotenv_path)
        job_utils.update_status(key, job_utils.JobStatus.FINISHED)
    except:
        job_utils.update_status(key, job_utils.JobStatus.ERROR)
