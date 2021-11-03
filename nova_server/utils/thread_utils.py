import threading
import logging
from threading import Thread

ml_lock = threading.Lock()
jc_lock = threading.Lock()
job_counter = 0


def ml_thread_wrapper(func):
    """
    Executing the function in a mutex protected thread for asynchronous execution of long running ml tasks
    :param func:
    :return: The thread id
    """

    def wrapper(*args, **kwargs):
        global job_counter

        def lock(*args, **kwargs):
            try:
                ml_lock.acquire()
                func(*args, **kwargs)
            finally:
                ml_lock.release()

        jc_lock.acquire()
        job_id = str(job_counter)
        job_counter += 1
        jc_lock.release()

        t = Thread(target=lock, name=job_id, args=args, kwargs=kwargs)
        t.start()

        return t.name

    return wrapper
