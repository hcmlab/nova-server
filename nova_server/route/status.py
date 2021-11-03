from flask import Blueprint, request, jsonify, send_file
from nova_server.utils.status_utils import JOBS, get_log_path, get_all_jobs

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

@status.route("/download/log/<id>", methods=["POST", "GET"])
def download_log(id):
    lp = get_log_path(id)
    if lp is None:
        job_status = {"error": "Unknown job id {}".format(id)}
        return jsonify(job_status)
    return send_file(lp)

@status.route("/jobstatusall", methods=["GET"])
def jobstatusall():
    return jsonify(get_all_jobs())
