"""Route to return the logfile for a specific job
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 13.09.2023
"""

from flask import Blueprint, request, jsonify
from nova_server.utils import log_utils
from nova_server.utils.job_utils import get_job_id_from_request_form


log = Blueprint("log", __name__)


@log.route("/log", methods=["POST"])
def log_thread():
    if request.method == "POST":
        request_form = request.form.to_dict()
        log_key = get_job_id_from_request_form(request_form)
        if log_key in log_utils.LOGS:
            logger = log_utils.LOGS[log_key]
            path = logger.handlers[0].baseFilename
            with open(path) as f:
                f = f.readlines()
            output = ""
            for line in f:
                output += line
            return jsonify({"message": output})
        else:
            return jsonify({"message": "No log for the given parameters found."})
