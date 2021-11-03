import threading
from threading import Thread

ml_lock = threading.Lock()

def ml_thread_wrapper(func):
    """
    Executing the function in a mutex protected thread for asynchronous execution of long running ml tasks
    :param func:
    :return: The thread id
    """
    def wrapper(*args, **kwargs):
        def lock(*args, **kwargs):
            try:
                ml_lock.acquire()
                func(*args, **kwargs)
            finally:
                ml_lock.release()
        t = Thread(target=lock, args=args, kwargs=kwargs)
        t.start()
        id = t.getName()
        return id
    return wrapper
