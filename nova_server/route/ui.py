from flask import Blueprint, render_template
from nova_server.utils import status_utils

ui = Blueprint("ui", __name__)

@ui.route('/')
def index():
    jobs = status_utils.get_all_jobs()
    return render_template('index.html', title='Current Jobs', jobs=jobs)
