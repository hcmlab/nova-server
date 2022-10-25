import logging
import threading
import pathlib
from nova_server.utils import status_utils


LOGS = {}


def get_logfile_name_for_thread(log_key):
    return log_key + "-job_" + threading.current_thread().name


def get_log_path_for_thread(log_key):
    name = get_logfile_name_for_thread(log_key)
    # TODO: add  dynamic log path from config
    return pathlib.Path(__file__).parent.parent / "logs" / (name + ".log")


def init_logger(logger, log_key):
    print("Init logger" + str(threading.current_thread().name))
    try:
        log_path = get_log_path_for_thread(log_key)
        status_utils.set_log_path(log_key, log_path)
        handler = logging.FileHandler(log_path, "w")
        handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        LOGS[log_key] = logger
        return logger
    except Exception as e:
        print(
            "Logger for {} could not be initialized.".format(
                str(threading.current_thread().name)
            )
        )
        raise e


def get_logger_for_thread(log_key):
    logger = logging.getLogger(get_logfile_name_for_thread(log_key))
    if not logger.handlers:
        logger = init_logger(logger, log_key)
    return logger


def remove_log_from_dict(log_key):
    LOGS.pop(log_key)
