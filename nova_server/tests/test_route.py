import threading
import logging
from nova_server.utils import log_utils
from flask import Blueprint, request, jsonify
from nova_server.utils import thread_utils, status_utils


test = Blueprint("test", __name__)

@test.route("/testjobqueque", methods=["POST"])
def test_jobqueque():
    if request.method == "POST":
        id = add_job()
        status_utils.add_new_job(id)
        data = {"job_id": id}
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def add_job():

    import time
    import random
    # Init logging
    logger = log_utils.get_logger_for_thread(__name__)
    logger.info('HERE IIII COOOOOME!')

    status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.RUNNING)
    total_time = random.randint(1, 250)
    for i in range(1, 11):
        current_progress = i * total_time // 10
        status_utils.update_progress(threading.current_thread().name, f"{current_progress} / {total_time}")
        time.sleep(total_time // 10)

    status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.FINISHED)
    status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.FINISHED)

