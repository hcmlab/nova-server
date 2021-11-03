from datetime import datetime
from enum import Enum

JOBS = {}


class JobStatus(Enum):
    RUNNING = 0
    FINISHED = 1
    ERROR = 2


class Job:
    def __init__(self, job_id, interactive_url=None):
        self.start_time = str(datetime.now())
        self.end_time = None
        self.progress = None
        self.status = JobStatus.RUNNING
        self.job_id = job_id
        self.interactive_url = interactive_url

    def serializable(self):
        s = vars(self)
        for key in s.keys():
            s[key] = str(s[key])
        return s

def add_new_job(job_id, interactive_url=None):
    job = Job(job_id, interactive_url)
    JOBS[job_id] = job
    return True


def remove_job(job_id):
    try:
        del JOBS[job_id]
    except KeyError:
        print(f"Key {job_id} is not in the dictionary")


def update_status(job_id, status: JobStatus):
    try:
        JOBS[job_id].status = status
    except KeyError:
        print(f"Key {job_id} is not in the dictionary")


def update_progress(job_id, progress: str):
    try:
        JOBS[job_id].progress = progress
    except KeyError:
        print(f"Key {job_id} is not in the dictionary")

def get_all_jobs():
    return [job.serializable() for job in JOBS.values()]
