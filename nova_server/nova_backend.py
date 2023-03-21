from flask import Flask
from nova_server.route.train import train
from nova_server.route.extract import extract
from nova_server.route.status import status
from nova_server.route.log import log
from nova_server.route.ui import ui
from nova_server.route.cancel import cancel
from nova_server.route.predict import predict
import argparse
from pathlib import Path

def create_app(template_folder):
    print("Starting nova-backend server")
    app = Flask(__name__, template_folder=template_folder)
    app.register_blueprint(train)
    app.register_blueprint(predict)
    app.register_blueprint(extract)
    app.register_blueprint(log)
    app.register_blueprint(status)
    app.register_blueprint(ui)
    app.register_blueprint(cancel)
    print("... done!")
    return app


if __name__ == "__main__":
    from waitress import serve

    parser = argparse.ArgumentParser(
        description="Commandline arguments to configure the nova backend server"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="The host ip address"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="The port the server listens on"
    )

    parser.add_argument(
        "--template_folder",
        type=str,
        default="./templates",
        help="Path for the templates to load relative to this script",
    )

    parser.add_argument(
        "--cml_dir",
        type=str,
        default="./cml",
        help="Cml folder to read the training scripts from. Same as in Nova.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Data folder to read the training scripts from. Same as in Nova.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Cache folder where all large files (e.g. model weights) are cached.",
    )

    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="./tmp",
        help="Folder for temporary data storage.",
    )

    # TODO: support multiple (data) directories
    args = parser.parse_args()
    app = create_app(template_folder=args.template_folder)
    app.config['CML_DIR'] = args.cml_dir
    app.config['DATA_DIR'] = args.data_dir
    app.config['CACHE_DIR'] = args.cache_dir
    app.config['TMP_DIR'] = args.tmp_dir

    Path(app.config['CACHE_DIR']).mkdir(parents=False, exist_ok=True)
    Path(app.config['TMP_DIR']).mkdir(parents=False, exist_ok=True)

    host = args.host
    port = args.port
    serve(app, host=host, port=port)
