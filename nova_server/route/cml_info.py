""" Blueprint for querying server capabilities

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    14.09.2023

This module defines a Flask Blueprint for querying the server for its capabilities.
Returns a list of all available server-module trainer and chains in json format.

"""
import os
import json
from flask import Blueprint, request
from nova_server.utils import env
from nova_utils.utils.json_utils import ChainEncoder, TrainerEncoder
from nova_utils.utils.ssi_xml_utils import Chain, Trainer
from pathlib import Path
from glob import glob

cml_info = Blueprint("cml_info", __name__)


@cml_info.route("/cml_info", methods=["GET"])
def info():
    """
    Query the server to return information about all

    This route allows querying the server for all available nova-server modules

    Returns:
        dict: A JSON response containing the status of the requested job.

    Example:

    """
    if request.method == "GET":
        cml_dir = os.getenv(env.NOVA_SERVER_CML_DIR)

        if not cml_dir:
            return None

        cml_dir = str(Path(cml_dir).resolve())

        trainer_files = glob(cml_dir + '/**/*.trainer', recursive = True)
        chain_files = glob(cml_dir + '/**/*.chain', recursive = True)

        trainer_ok = {}
        trainer_faulty = {}
        chains_ok = {}
        chains_faulty = {}

        for tf in trainer_files:
            t = Trainer()
            rtf = str(Path(tf).relative_to(cml_dir))
            try:
                t.load_from_file(tf)
                trainer_ok[rtf] = json.dumps(t, cls=TrainerEncoder)
            except Exception as e:
                trainer_faulty[rtf] = str(e)

        for cf in chain_files:
            c = Chain()
            rcf = str(Path(cf).relative_to(cml_dir))
            try:
                c.load_from_file(cf)
                chains_ok[rcf] = json.dumps(c, cls=ChainEncoder)
            except Exception as e:
                chains_faulty[rcf] = str(e)

        return {
            'chains_ok' : chains_ok,
            'chains_faulty' : chains_faulty,
            'trainer_ok' : trainer_ok,
            'trainer_faulty' : trainer_faulty
        }



