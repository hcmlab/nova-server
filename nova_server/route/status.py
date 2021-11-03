from flask import Blueprint, request, jsonify
from nova_server.utils.status_utils import JOBS

status = Blueprint("status", __name__)

@status.route("/jobstatus", methods=["POST"])
def jobstatus():
    if request.method == "POST":
        id = request.form.get("job_id")
        if id in JOBS.keys():
            job_status = JOBS[id].serializable()
        else:
            job_status = {"error": "Unknown job id {}".format(id)}

        return jsonify(job_status)

