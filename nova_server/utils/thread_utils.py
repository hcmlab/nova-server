import threading
from threading import Thread

status_lock = threading.Lock()
ml_lock = threading.Lock()
jc_lock = threading.Lock()
job_counter = 0


def ml_thread_wrapper(func):
    """
    Executing the function in a mutex protected thread for asynchronous execution of long running ml tasks.
    The thread waits with execution unit the status_lock is released. To do any initialization before the thread starts acquire the status lock before creating the thread.
    :param func:
    :return: The thread
    """

    def wrapper(*args, **kwargs):
        global job_counter

        def lock(*args, **kwargs):
            try:
                # Only one ml thread can be active at the same time
                ml_lock.acquire()
                func(*args, **kwargs)
            finally:
                ml_lock.release()

        jc_lock.acquire()
        job_id = str(job_counter)
        job_counter += 1
        jc_lock.release()

        t = Thread(target=lock, name=job_id, args=args, kwargs=kwargs)

        return t

    return wrapper


def status_thread_wrapper(func):
    def wrapper(*args, **kwargs):
        try:
            status_lock.acquire()
            return func(*args, **kwargs)
        finally:
            status_lock.release()

    return wrapper
