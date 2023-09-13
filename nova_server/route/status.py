"""Route to query the current status of a specific job or all logged jobs
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 13.09.2023
"""

from flask import Blueprint, request, jsonify
from nova_server.utils.job_utils import JOBS, get_all_jobs, JobStatus
from nova_server.utils.job_utils import get_job_id_from_request_form

status = Blueprint("status", __name__)


@status.route("/job_status", methods=["POST"])
def job_status():
    if request.method == "POST":
        request_form = request.form.to_dict()
        status_key = get_job_id_from_request_form(request_form)

        if status_key in JOBS.keys():
            status = JOBS[status_key].status
            return jsonify({"status": status.value})
        else:
            return jsonify({"status": JobStatus.WAITING.value})


@status.route("/job_status_all", methods=["GET"])
def job_status_all():
    return jsonify(get_all_jobs())
