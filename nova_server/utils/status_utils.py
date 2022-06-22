from datetime import datetime
from enum import Enum
from nova_server.utils.thread_utils import status_thread_wrapper

JOBS = {}


class JobStatus(Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2
    ERROR = 3


class Job:
    def __init__(self, job_id, job_type=None, interactive_url=None, log_path=None):
        self.start_time = None
        self.end_time = None
        self.progress = None
        self.status = JobStatus.WAITING
        self.job_id = job_id
        self.job_type = job_type
        self.interactive_url = interactive_url
        self.log_path = log_path

    def serializable(self):
        s = vars(self)
        for key in s.keys():
            s[key] = str(s[key])
        return s


@status_thread_wrapper
def add_new_job(job_id, job_type, interactive_url=None):
    job = Job(job_id, job_type, interactive_url)
    JOBS[job_id] = job
    return True


@status_thread_wrapper
def remove_job(job_id):
    try:
        del JOBS[job_id]
    except KeyError:
        print(f"Key {job_id} is not in the dictionary")


@status_thread_wrapper
def update_status(job_id, status: JobStatus):
    try:
        JOBS[job_id].status = status

        if status == status.RUNNING:
            JOBS[job_id].start_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        if status == status.FINISHED or status == status.ERROR:
            JOBS[job_id].end_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    except KeyError:
        print(f"Key {job_id} is not in the dictionary")


@status_thread_wrapper
def update_progress(job_id, progress: str):
    try:
        JOBS[job_id].progress = progress
    except KeyError:
        print(f"Key {job_id} is not in the dictionary")


@status_thread_wrapper
def set_log_path(job_id, log_path):
    try:
        JOBS[job_id].log_path = log_path
    except KeyError:
        print(f"Key {job_id} is not in the dictionary")


@status_thread_wrapper
def get_log_path(job_id):
    try:
        return JOBS[str(job_id)].log_path
    except KeyError:
        print(f"Key {job_id} is not in the dictionary")


@status_thread_wrapper
def get_all_jobs():
    return [job.serializable() for job in JOBS.values()]
