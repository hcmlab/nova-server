from flask import Blueprint, request, jsonify
from nova_server.utils import log_utils
from nova_server.utils.key_utils import get_key_from_request_form


log = Blueprint("log", __name__)


@log.route("/log", methods=["POST"])
def complete_thread():
    if request.method == "POST":
        log_key = get_key_from_request_form(request.form)
        if log_key in log_utils.LOGS:
            logger = log_utils.LOGS[log_key]
            path = logger.handlers[0].baseFilename
            with open(path) as f:
                f = f.readlines()
            output = ''
            for line in f:
                output += line
            return jsonify({'message': output})
        else:
            return jsonify({'message': ''})
