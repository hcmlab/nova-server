""" Blueprint for retrieving a job's log file

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    27.10.2023

This module defines a Flask Blueprint for retrieving a job's log file.

"""
import os

from flask import Blueprint, request, jsonify
from nova_server.utils.job_utils import get_job_id_from_request_form
from nova_server.utils import env

fetch_result = Blueprint("fetch_result", __name__)


@fetch_result.route("/fetch_result", methods=["POST"])
def fetch_thread():
    """
    Retrieve the results of a specific job.

    This route allows retrieving the data file for a job by providing the job's unique identifier in the request.

    Returns:
        dict: Data object for the respective job. 404 if not data has been found

    Example:
        >>> POST /log
        >>> {"job_id": "12345"}
        {"message": "Log file content here..."}
    """
    if request.method == "POST":
        request_form = request.form.to_dict()
        job_id = get_job_id_from_request_form(request_form)
        tmp_dir = os.getenv(env.NOVA_SERVER_TMP_DIR)


