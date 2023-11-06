""" Blueprint for retrieving a job's log file

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    27.10.2023

This module defines a Flask Blueprint for retrieving a job's log file.

"""
import os

from flask import Blueprint, request, jsonify, after_this_request
from nova_server.utils.job_utils import get_job_id_from_request_form
from nova_server.utils import env
from pathlib import Path
import shutil
import tempfile
fetch_result = Blueprint("fetch_result", __name__)


import zipfile
import os
from flask import send_file,Flask,send_from_directory

def supported_file(x: Path):
    file_names = [
    ]
    file_starts_with = [
        '.'
    ]
    file_ext = [

    ]
    return  not(x.name in file_names or any([x.name.startswith(s) for s in file_starts_with]) or x.suffix in file_ext)

@fetch_result.route("/fetch_result", methods=["POST"])
def fetch_thread():
    """
    Retrieve the results of a specific job.

    This route allows retrieving the data file for a job by providing the job's unique identifier in the request.

    Returns:
        dict: Data object for the respective job. 404 if not data has been found

    Example:
        >>> POST /log
        >>> {"job_id": "12345"}
        {"message": "Log file content here..."}
    """
    if request.method == "POST":
        request_form = request.form.to_dict()
        job_id = get_job_id_from_request_form(request_form)

        shared_dir = os.getenv(env.NOVA_SERVER_TMP_DIR)
        job_dir = Path(shared_dir) / job_id

        if not job_dir.exists():
            raise FileNotFoundError
        elif job_dir.is_dir():

            files = list(filter(lambda x: supported_file(x) and x.is_file(), job_dir.rglob('*')))

            if len(files) > 1:
                # Zip file Initialization
                zip_fp = tempfile.TemporaryFile()
                zipfolder = zipfile.ZipFile(zip_fp,'w', compression = zipfile.ZIP_STORED)
                for file in files:
                    zipfolder.write(file, arcname=file.relative_to(job_dir.parent))
                zipfolder.close()

                # Reset file pointer to keep handle alive
                zip_fp.seek(0)

                return send_file(zip_fp,
                                 mimetype = 'zip',
                                 download_name= 'result.zip',
                               )
            elif len(files) == 1:
                return send_file(files[0])
            else:
                raise ValueError('Empty folder')
        else:
            return send_file(job_dir)


