import sys

sys.path.append('/Users/Marco/Documents/Uni/Masterarbeit/hcai_datasets')
from hcai_datasets.hcai_nova_dynamic.nova_db_handler import NovaDBHandler


def write_polygons_to_db(request_form, polygons, confidences):
    db_config_dict = {
        'ip': request_form["server"].split(':')[0],
        'port': int(request_form["server"].split(':')[1]),
        'user': request_form["username"],
        'password': request_form["password"]
    }

    db_handler = NovaDBHandler(db_config_dict=db_config_dict)

    database = request_form['database']
    scheme = request_form['scheme']
    session = request_form['sessions']
    annotator = request_form['annotator']
    role = request_form['roles']
    start = int(int(request_form['cmlBeginTime']) / (float(request_form['sampleRate']) * 1000))

    return db_handler.save_polygons(database, scheme, session, annotator, role, polygons, confidences, start)
