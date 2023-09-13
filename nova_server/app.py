import dotenv
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
from nova_server.utils import env

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

def run():

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
        args.cml_dir, env.NOVA_SERVER_CML_DIR, default_args.cml_dir, create_directory=True
    )
    os.environ[env.NOVA_SERVER_DATA_DIR] = resolve_arg(
        args.data_dir, env.NOVA_SERVER_DATA_DIR, default_args.data_dir, create_directory=True
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
    print("...done")

    host = os.environ[env.NOVA_SERVER_HOST]
    port = int(os.environ[env.NOVA_SERVER_PORT])

    serve(app, host=host, port=port)

if __name__ == '__main__':
    run()
