import threading

from flask import Blueprint, request, jsonify
from nova_server.utils import thread_utils, status_utils, log_utils
from nova_server.utils.key_utils import get_key_from_request_form
from flask import current_app
from nova_server.utils.thread_utils import THREADS
from nova_server.utils.status_utils import update_progress


extract = Blueprint("extract", __name__)


@extract.route("/extract", methods=["POST"])
def extract_thread():
    if request.method == "POST":
        request_form = request.form.to_dict()
        key = get_key_from_request_form(request_form)
        thread = extract_data(request_form, current_app._get_current_object())
        status_utils.add_new_job(key)
        data = {"success": "true"}
        thread.start()
        THREADS[key] = thread
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def extract_data(request_form, app_context):
    key = get_key_from_request_form(request_form)
    logger = log_utils.get_logger_for_thread(key)
    cml_dir = app_context.config["CML_DIR"]
    data_dir = app_context.config["DATA_DIR"]

    # TODO chain parser
    chain_file_path = Path(cml_dir).joinpath(
        PureWindowsPath(request_form["trainerFilePath"])
    )

    logger.info("Action 'Extract' started.")



    status_utils.update_status(key, status_utils.JobStatus.RUNNING)
    sessions = request_form["sessions"].split(";")
    roles = request_form["roles"].split(";")



    trainer_file_path = Path(cml_dir).joinpath(
        PureWindowsPath(request_form["trainerFilePath"])
    )
    trainer = Trainer()

    update_progress("Initalizing")
    logger = log_utils.get_logger_for_thread(__name__)
    log_conform_request = dict(request_form)
    log_conform_request["password"] = "---"
    logger.info(f"Start Extracting with request {log_conform_request}")

    logger.error("Not implemented.")

    update_progress("Done")
    status_utils.update_status(
        threading.current_thread().name, status_utils.JobStatus.ERROR
    )
    return
