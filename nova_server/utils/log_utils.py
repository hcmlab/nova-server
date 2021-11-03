import logging
import threading
import pathlib
from nova_server.utils import status_utils


def get_logfile_name_for_thread(module_name):
    return module_name + "-job_" + threading.current_thread().name


def get_log_path_for_thread(module_name):
    name = get_logfile_name_for_thread(module_name)
    # TODO: add  dynamic log path from config
    return pathlib.Path(__file__).parent.parent / "logs" / (name + ".log")

def init_logger(logger, module_name):
    log_path = get_log_path_for_thread(module_name)
    job_id = threading.current_thread().name

    if not job_id in status_utils.JOBS.keys():
        status_utils.add_new_job(job_id)

    status_utils.JOBS[job_id].log_path = log_path
    handler = logging.FileHandler(log_path, "w")
    handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

def get_logger_for_thread(module_name):
    logger = logging.getLogger(get_logfile_name_for_thread(module_name))
    if not logger.handlers:
        logger = init_logger(logger, module_name)
    return logger