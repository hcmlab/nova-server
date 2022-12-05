import sys

sys.path.append('/Users/Marco/Documents/Uni/Masterarbeit/hcai_datasets')
from hcai_datasets.hcai_nova_dynamic.nova_db_handler import NovaDBHandler
from hcai_datasets.hcai_nova_dynamic.utils import nova_data_utils

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

def write_freeform_to_db(request_form, results: dict):

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

def write_discrete_to_db(request_form, results: list):
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
    roles = request_form['roles']

    frame_size = nova_data_utils.parse_time_string_to_ms(request_form['frameSize'])

    annos = []
    last_label = None
    current_label_start = 1
    for i, x in enumerate(results):
        # current label is different from the one before
        if not x == last_label and last_label is not None:
            frame_from = str((current_label_start * frame_size) / 1000.0)
            frame_to = str((i * frame_size) / 1000.0)

            annos.append({
                'from': frame_from,
                'to': frame_to,
                'conf': 1.0,
                'id': int(last_label)
            })
            current_label_start = i
        last_label = x


    # TODO: We only take one role into account in this case. Fix
    role = roles.split(';')[0]
    db_handler.set_annos(
        dataset=database,
        scheme=scheme,
        session=session,
        annotator=annotator,
        role=role,
        annos=annos,
    )