from flask import Flask
from nova_server.route.train import train
from nova_server.route.extract import extract
from nova_server.route.status import status
from nova_server.route.log import log
from nova_server.route.ui import ui
from nova_server.route.cancel import cancel
from nova_server.route.predict import predict
import argparse
import os
from pathlib import Path
from waitress import serve


print("Starting nova-backend server...")
app = Flask(__name__, template_folder="./templates")
app.register_blueprint(train)
app.register_blueprint(predict)
app.register_blueprint(extract)
app.register_blueprint(log)
app.register_blueprint(status)
app.register_blueprint(ui)
app.register_blueprint(cancel)

parser = argparse.ArgumentParser(
    description="Commandline arguments to configure the nova backend server"
)
parser.add_argument("--host", type=str, default="0.0.0.0", help="The host ip address")
parser.add_argument(
    "--port", type=int, default=8080, help="The port the server listens on"
)

parser.add_argument(
    "--template_folder",
    type=str,
    default="templates",
    help="Path for the templates to load relative to this script",
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


# TODO: support multiple (data) directories
args = parser.parse_args()
default_args = parser.parse_args([])

host = args.host
port = args.port
print(f"\tHOST: {host}\n\tPORT: {port}")

# setting environment variables in the following priority from highest to lowest:
# provided argument
# environment variable
# default value
def get_dir_from_arg(arg_val, env_var, arg_default_val, create_if_not_exist=True):
    val = None
    # check if argument has been provided
    if not arg_val == arg_default_val:
        val = arg_val
    # check if environment variable exists
    elif os.environ.get(env_var):
        val = os.environ[env_var]
    # return default
    else:
        val = arg_default_val
    print(f"\t{env_var} : {val}")
    Path(val).mkdir(parents=False, exist_ok=True)
    return val


os.environ["NOVA_CML_DIR"] = get_dir_from_arg(
    args.cml_dir, "NOVA_CML_DIR", default_args.cml_dir
)
os.environ["NOVA_DATA_DIR"] = get_dir_from_arg(
    args.data_dir, "NOVA_DATA_DIR", default_args.data_dir
)
os.environ["NOVA_CACHE_DIR"] = get_dir_from_arg(
    args.cache_dir, "NOVA_CACHE_DIR", default_args.cache_dir
)
os.environ["NOVA_TMP_DIR"] = get_dir_from_arg(
    args.tmp_dir, "NOVA_TMP_DIR", default_args.tmp_dir
)
os.environ["NOVA_LOG_DIR"] = get_dir_from_arg(
    args.log_dir, "NOVA_LOG_DIR", default_args.log_dir
)
print("...done")

serve(app, host=host, port=port)
