from flask import Blueprint, request, jsonify
from nova_server.route.train import train_model
from nova_server.route.predict import predict_data
from nova_server.utils.thread_utils import THREADS
from nova_server.utils import thread_utils, status_utils, log_utils
from nova_server.utils.key_utils import get_key_from_request_form

complete = Blueprint("complete", __name__)
thread = Blueprint("thread", __name__)


@complete.route("/complete", methods=["POST"])
def complete_thread():
    if request.method == "POST":
        request_form = request.form.to_dict()
        key = get_key_from_request_form(request_form)
        thread = complete_session(request_form)
        status_utils.add_new_job(key)
        data = {"success": "true"}
        thread.start()
        THREADS[key] = thread
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def complete_session(request_form):
    weights_path = train_model(request_form)

    if weights_path is None:
        key = get_key_from_request_form(request_form)
        logger = log_utils.get_logger_for_thread(key)
        logger.error("An error occurred while training!")
        return

    request_form["weightsPath"] = weights_path
    predict_data(request_form)
