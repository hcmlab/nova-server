from flask import Blueprint, render_template
from nova_server.utils import status_utils

ui = Blueprint("ui", __name__)

@ui.route('/')
def index():
    jobs = status_utils.get_all_jobs()
    return render_template('ajax_template.html', title='Current Jobs', jobs=jobs)

@ui.route('/data')
def data():
    return {'data': status_utils.get_all_jobs()}
