import nova_server.utils.status_utils

from flask import Blueprint, request, jsonify
from nova_server.utils import status_utils
from nova_server.utils.key_utils import get_key_from_request_form
from nova_server.utils.thread_utils import THREADS
from nova_server.utils.log_utils import LOGS


cancel = Blueprint("cancel", __name__)


@cancel.route("/cancel", methods=["POST"])
def complete_thread():
    if request.method == "POST":
        request_form = request.form.to_dict()
        key = get_key_from_request_form(request_form)

        if key in THREADS:
            thread = THREADS[key]
            thread.raise_exception()
            status_utils.update_status(key, status_utils.JobStatus.WAITING)
            if key in LOGS:
                log = LOGS[key]
                log.info("Action successfully canceled.")
            return jsonify({'success': "true"})
        else:
            if key in LOGS:
                log = LOGS[key]
                log.info("Cancel was not successful.")
            return jsonify({'success': "false"})
