from flask import Flask
from nova_server.route.train import train
from nova_server.route.status import status
from nova_server.route.log import log
from nova_server.route.ui import ui
from nova_server.route.cancel import cancel
from nova_server.route.predict import predict
from nova_server.route.complete import complete


def create_server():
    print("Starting nova-backend server...")
    app = Flask(__name__)
    app.register_blueprint(train)
    app.register_blueprint(predict)
    app.register_blueprint(complete)
    app.register_blueprint(log)
    app.register_blueprint(status)
    app.register_blueprint(ui)
    app.register_blueprint(cancel)
    print("... done!")
    return app


if __name__ == "__main__":
    from waitress import serve

    app = create_server()
    serve(app, host="localhost", port=27017)
