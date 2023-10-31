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
fetch_result = Blueprint("fetch_result", __name__)


import zipfile
import os
from flask import send_file,Flask,send_from_directory


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
        tmp_dir = Path(os.getenv(env.NOVA_SERVER_TMP_DIR))

        shared_dir = os.getenv(env.NOVA_SERVER_TMP_DIR)
        job_dir = Path(shared_dir) / job_id

        if not job_dir.exists():
            raise FileNotFoundError
        elif job_dir.is_dir():
            # Zip file Initialization
            zip_fp = tmp_dir / 'tmp.zip'
            zipfolder = zipfile.ZipFile(zip_fp,'w', compression = zipfile.ZIP_STORED) # Compression type

            # zip all the files which are inside in the folder
            for files in job_dir.glob('*'):
                for file in files:
                    zipfolder.write(file, file.name)
            zipfolder.close()

            # Delete the zip file if not needed
            @after_this_request
            def delete_file():
                os.remove(zip_fp)

            return send_file(zip_fp,
                             mimetype = 'zip',
                             attachment_filename= 'tmp.zip',
                             as_attachment = True)
        else:
            return send_file(job_dir)


