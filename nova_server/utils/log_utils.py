"""Utility modules for NOVA-Server Logs

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    13.09.2023

"""

import logging
import threading
from pathlib import Path
import os
from nova_server.utils import job_utils, env

LOGS = {}


def get_log_conform_request(request_form):
    log_conform_request = dict(request_form)
    log_conform_request["password"] = "---"
    return log_conform_request


def get_log_path_for_thread(job_id):
    log_dir = os.environ[env.NOVA_SERVER_LOG_DIR]
    return Path(log_dir) / (job_id + ".log")


def init_logger(logger, job_id):
    print("Init logger" + str(threading.current_thread().name))
    try:
        log_path = get_log_path_for_thread(job_id)
        job_utils.set_log_path(job_id, log_path)
        handler = logging.FileHandler(log_path, "w")
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        LOGS[job_id] = logger
        return logger
    except Exception as e:
        print(
            "Logger for {} could not be initialized.".format(
                str(threading.current_thread().name)
            )
        )
        raise e


def get_logger_for_job(job_id):
    logger = logging.getLogger(job_id)
    if not logger.handlers:
        logger = init_logger(logger, job_id)
    return logger


def remove_log_from_dict(job_id):
    LOGS.pop(job_id)
