from flask import Flask
from nova_server.route.train import train

print("Starting nova-backend server")
app = Flask(__name__)
app.register_blueprint(train)

if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=8080)

# TODO: Init database access
# app.config['MONGODB_SETTINGS'] = {
# 'host': 'mongodb://localhost/movie-bag'
# }
