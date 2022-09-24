from flask import Blueprint, request, jsonify
from nova_server.utils.key_utils import get_key_from_request_form


cancel = Blueprint("cancel", __name__)


@cancel.route("/cancel", methods=["POST"])
def complete_thread():
    if request.method == "POST":
        thread_key = get_key_from_request_form(request.form)

        # TODO
        # Check if Thread is available with the given thread_key
        # If cancel thread and write into log, that it was successful
        # else write into log, that it was not successful
        return jsonify({'success': "false"})
