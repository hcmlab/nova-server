import threading

from flask import Blueprint, request, jsonify
from nova_server.utils import thread_utils, status_utils, log_utils

extract = Blueprint("extract", __name__)


@extract.route("/extract", methods=["POST"])
def extract_thread():
    if request.method == "POST":
        thread = extract_data(request.form)
        thread_id = thread.name
        status_utils.add_new_job(thread_id, extract.name)
        data = {"job_id": thread_id}
        thread.start()
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def extract_data(request_form):
    def update_progress(msg):
        status_utils.update_progress(threading.current_thread().name, msg)

    # Init logging
    status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.RUNNING)

    update_progress('Initalizing')
    logger = log_utils.get_logger_for_thread(__name__)
    log_conform_request = dict(request_form)
    log_conform_request['password'] = '---'
    logger.info(f"Start Extracting with request {log_conform_request}")

    logger.error('Not implemented.')

    update_progress('Done')
    status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.ERROR)
    return

