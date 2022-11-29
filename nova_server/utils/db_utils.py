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
    start_frame = int(int(request_form['startTime']) / int(request_form['frameSize']))

    return db_handler.save_polygons(database, scheme, session, annotator, role, polygons, confidences, start_frame)

def write_freeform_to_db(request_form, results):

    # TODO check if we really need to establish a new connection to the database
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
    #role = request_form['roles']

    annos = {}
    for frame, results in results.items():
        frame_info = frame.split('_')
        frame_from = float(frame_info[-2])
        frame_to = float(frame_info[-1])
        for stream_id, anno in results.items():

            conf = anno['conf']
            name = anno['name']

            if not stream_id in annos.keys():
                annos[stream_id] = []

            annos[stream_id].append({
                "from": frame_from,
                "to": frame_to,
                "conf": conf,
                "name": name
            })



    for anno_id, anno in annos.items():
        # TODO does not work with flattened roles
        role, stream = anno_id.split('.')

        db_handler.set_annos(
            dataset=database,
            scheme=scheme,
            session=session,
            annotator=annotator,
            role=role,
            annos=anno,
        )