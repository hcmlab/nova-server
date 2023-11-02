"""Standalone script for data predictions

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    06.09.2023

This script performs predictions for annotations in the NOVA database using a provided nova-server module for inference and saves the results to the NOVA-DB.

Args:
    --env (str, optional): Path to the environment file to read configuration from. Default: ''
    --host (str, optional): The host IP address for the server. Default: '0.0.0.0'
    --port (str, optional): The port on which the server will listen. Default: '8080'
    --cml_dir (str, optional): CML folder to read the training scripts from, same as in Nova. Default: 'cml'
    --data_dir (str, optional): Data folder to read the training scripts from, same as in Nova. Default: 'data'
    --cache_dir (str, optional): Cache folder where all large files (e.g., model weights) are cached. Default: 'cache'
    --tmp_dir (str, optional): Folder for temporary data storage. Default: 'tmp'
    --log_dir (str, optional): Folder for temporary data storage. Default: 'log'

Returns:
    None

Example:
    >>> app.py --host 0.0.0.0 --port 53771 --cml_dir "/path/to/my/cml" --data_dir "/path/to/my/data" --cache_dir "/path/to/my/cache" --tmp_dir "/path/to/my/tmp"
"""
import dotenv
import tempfile
from flask import Flask
from nova_server.route.train import train
from nova_server.route.status import status
from nova_server.route.log import log
from nova_server.route.ui import ui
from nova_server.route.cml_info import cml_info
from nova_server.route.cancel import cancel
from nova_server.route.process import process
from nova_server.route.fetch_result import fetch_result
import argparse
import os
from pathlib import Path
from waitress import serve
from nova_server.utils import env

print("Starting nova-backend server...")
app = Flask(__name__, template_folder="./templates")
app.register_blueprint(train)
app.register_blueprint(process)
app.register_blueprint(log)
app.register_blueprint(status)
app.register_blueprint(ui)
app.register_blueprint(cancel)
app.register_blueprint(cml_info)
app.register_blueprint(fetch_result)

parser = argparse.ArgumentParser(
    description="Commandline arguments to configure the nova backend server"
)
parser.add_argument("--env", type=str, default='', help="Path to the environment file to read config from")
parser.add_argument("--host", type=str, default="0.0.0.0", help="The host ip address")
parser.add_argument(
    "--port", type=str, default='8080', help="The port the server listens on"
)
parser.add_argument(
    "--cml_dir",
    type=str,
    default="cml",
    help="Cml folder to read the training scripts from. Same as in Nova.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="data",
    help="Data folder to read the training scripts from. Same as in Nova.",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    default="cache",
    help="Cache folder where all large files (e.g. model weights) are cached.",
)

parser.add_argument(
    "--tmp_dir",
    type=str,
    default="tmp",
    help="Folder for temporary data storage.",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default="log",
    help="Folder for temporary data storage.",
)

parser.add_argument(
    "--backend",
    type=str,
    default="backend",
    help="The backend used for processing requests",
)

def _run():

    # TODO: support multiple (data) directories
    args = parser.parse_args()
    default_args = parser.parse_args([])

    # Loading dot env file if provided
    if args.env:
        env_path = Path(args.env)
        if env_path.is_file():
            print(f'loading environment from {env_path.resolve()}')
            dotenv.load_dotenv(env_path, verbose=True, override=True)
        else:
            raise FileNotFoundError(f'.env file not found at {env_path} ')

    # Setting environment variables in the following priority from highest to lowest:
    # Provided commandline argument -> Dotenv environment variable -> System environment variable -> Default value

    def resolve_arg(arg_val, env_var, arg_default_val, create_directory=False):
        # Check if argument has been provided
        if not arg_val == arg_default_val:
            val = arg_val
        # Check if environment variable exists
        elif os.environ.get(env_var):
            val = os.environ[env_var]
        # Return default
        else:
            val = arg_default_val
        print(f"\t{env_var} : {val}")

        if create_directory:
            Path(val).mkdir(parents=False, exist_ok=True)

        return val

    # Write values to OS variables
    os.environ[env.NOVA_SERVER_HOST] = resolve_arg(
        args.host, env.NOVA_SERVER_HOST, default_args.host
    )
    os.environ[env.NOVA_SERVER_PORT] = resolve_arg(
        args.port, env.NOVA_SERVER_PORT, default_args.port
    )

    os.environ[env.NOVA_SERVER_CML_DIR] = resolve_arg(
        args.cml_dir, env.NOVA_SERVER_CML_DIR, default_args.cml_dir
    )
    os.environ[env.NOVA_SERVER_DATA_DIR] = resolve_arg(
        args.data_dir, env.NOVA_SERVER_DATA_DIR, default_args.data_dir
    )
    os.environ[env.NOVA_SERVER_CACHE_DIR] = resolve_arg(
        args.cache_dir, env.NOVA_SERVER_CACHE_DIR, default_args.cache_dir, create_directory=True
    )
    os.environ[env.NOVA_SERVER_TMP_DIR] = resolve_arg(
        args.tmp_dir, env.NOVA_SERVER_TMP_DIR, default_args.tmp_dir, create_directory=True
    )
    os.environ[env.NOVA_SERVER_LOG_DIR] = resolve_arg(
        args.log_dir, env.NOVA_SERVER_LOG_DIR, default_args.log_dir, create_directory=True
    )
    os.environ[env.NOVA_SERVER_BACKEND] = resolve_arg(
        args.backend, env.NOVA_SERVER_BACKEND, default_args.backend
    )

    print("...done")
    tempfile.tempdir = os.environ[env.NOVA_SERVER_TMP_DIR]
    host = os.environ[env.NOVA_SERVER_HOST]
    port = int(os.environ[env.NOVA_SERVER_PORT])

    serve(app, host=host, port=port, threads=8)

if __name__ == '__main__':
    _run()
