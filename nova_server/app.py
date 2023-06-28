import os
from flask import Flask
from pathlib import Path
from waitress import serve
from nova_server.route.cancel import cancel
from nova_server.route.extract import extract
from nova_server.route.log import log
from nova_server.route.predict import predict
from nova_server.route.status import status
from nova_server.route.train import train
from nova_server.route.ui import ui


def build_server():
    """Builds a Flask server with registered blueprints.
     Returns:
        Flask: A Flask server instance with registered blueprints.
    """
    app = Flask(__name__, template_folder="./templates")
    app.register_blueprint(train)
    app.register_blueprint(predict)
    app.register_blueprint(extract)
    app.register_blueprint(log)
    app.register_blueprint(status)
    app.register_blueprint(ui)
    app.register_blueprint(cancel)
    return app


def setup_env_variable(key: str, value: str, default: str):
    """Sets up an environment variable with the provided value or default value.
     Args:
        key (str): The name of the environment variable to set.
        value (str): The value to set the environment variable to, if provided.
        default (str): The default value to set the environment variable to, if no value is provided.
     Raises:
        TypeError: If the key or default arguments are not strings.
     Examples:
        >>> setup_env_variable('NOVA_CML_DIR', 'my/cml/dir', 'cml')
        NOVA_CML_DIR : my/cml/dir
        >>> setup_env_variable('NOVA_DATA_DIR', None, 'data')
        NOVA_DATA_DIR : data
    """
    if not isinstance(key, str) or not isinstance(default, str):
        raise TypeError("Key and default arguments must be strings.")

    # check if argument has been provided
    if value and not value == default:
        val = value
    # check if environment variable exists
    elif os.environ.get(key):
        val = os.environ[key]
    # return default
    else:
        val = default
    print(f"\t{key} : {val}")
    Path(val).mkdir(parents=False, exist_ok=True)

    os.environ[key] = val


def main(host: str = '0.0.0.0', port: int = 8080, cml_dir: str = None, data_dir: str = None, cache_dir: str = None,
         tmp_dir: str = None, log_dir: str = None):
    """Sets up environment variables and starts the server.
     Args:
        host (str, optional): The hostname to listen on. Defaults to '0.0.0.0'.
        port (int, optional): The port of the web server. Defaults to 8080.
        cml_dir (str, optional): The directory to store CML files. Defaults to None.
        data_dir (str, optional): The directory to store data files. Defaults to None.
        cache_dir (str, optional): The directory to store cache files. Defaults to None.
        tmp_dir (str, optional): The directory to store temporary files. Defaults to None.
        log_dir (str, optional): The directory to store log files. Defaults to None.
     Raises:
        TypeError: If any of the directory arguments are not strings.
     Examples:
        >>> main()
        Starting nova-backend server...
        NOVA_CML_DIR : cml
        NOVA_DATA_DIR : data
        NOVA_CACHE_DIR : cache
        NOVA_TMP_DIR : tmp
        NOVA_LOG_DIR : log
        ...done
    """
    if not all(isinstance(dir, str) or dir is None for dir in [cml_dir, data_dir, cache_dir, tmp_dir, log_dir]):
        raise TypeError("Directory arguments must be strings or None.")
    # Initializing system environment variables
    setup_env_variable('NOVA_CML_DIR', cml_dir, 'cml')
    setup_env_variable('NOVA_DATA_DIR', data_dir, 'data')
    setup_env_variable('NOVA_CACHE_DIR', cache_dir, 'cache')
    setup_env_variable('NOVA_TMP_DIR', tmp_dir, 'tmp')
    setup_env_variable('NOVA_LOG_DIR', log_dir, 'log')
    # Starting server
    print("Starting nova-backend server...")
    app = build_server()
    serve(app, host=host, port=port)
    print("...done")
