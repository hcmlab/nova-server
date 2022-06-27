from flask import Flask
from nova_server.route.train import train
from nova_server.route.predict import predict
from nova_server.route.extract import extract
from nova_server.route.status import status
from nova_server.route.ui import ui
from nova_server.tests.test_route import test

import argparse

parser = argparse.ArgumentParser(description='Commandline arguments to configure the nova backend server')
parser.add_argument('--host', type=str, default='0.0.0.0', help='The host ip address')
parser.add_argument('--port', type=int, default=8080, help='The port the server listens on')
parser.add_argument('--template_folder', type=str, optional=True, default='.', help='Path for the templates to load relative to this script')


args = parser.parse_args()
print(args.accumulate(args.integers))


def create_app(template_folder):
    print("Starting nova-backend server")
    app = Flask(__name__, template_folder)
    app.register_blueprint(train)
    app.register_blueprint(predict)
    app.register_blueprint(extract)
    app.register_blueprint(status)
    app.register_blueprint(ui)
    app.register_blueprint(test)
    return app


if __name__ == "__main__":
    from waitress import serve
    app = create_app(template_folder=args.template_folder)
    host = args.host
    ip = args.ip
    serve(app, host=host, port=ip)
