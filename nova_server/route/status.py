from flask import Blueprint, request, jsonify
from nova_server.utils.status_utils import JOBS, get_all_jobs, JobStatus
from nova_server.utils.key_utils import get_key_from_request_form

status = Blueprint("status", __name__)


@status.route("/job_status", methods=["POST"])
def job_status():
    if request.method == "POST":
        request_form = request.form.to_dict()
        status_key = get_key_from_request_form(request_form)

        if status_key in JOBS.keys():
            status = JOBS[status_key].status
            return jsonify({"status": status.value})
        else:
            return jsonify({"status": JobStatus.WAITING.value})


@status.route("/job_status_all", methods=["GET"])
def job_status_all():
    return jsonify(get_all_jobs())
