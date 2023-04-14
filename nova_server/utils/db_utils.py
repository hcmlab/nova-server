import sys
import bson
import copy
import warnings
import numpy as np

from hcai_datasets.hcai_nova_dynamic.nova_db_handler import NovaDBHandler
from hcai_datasets.hcai_nova_dynamic.utils import nova_data_utils

MAX_MONGO_DB_DOC_SIZE = 16777216
database = None
scheme = None
session = None
annotator = None
roles = None


# Format of results must be: {values: [value1, value2, value3, ...], confidences: [conf1, conf2, conf3, ...], start_frame: x, ...}
def check_format(variable, logger):
    text = ""
    if type(variable) is not dict:
        text = 'Result type must be a dictionary!'
    if "values" not in variable or "confidences" not in variable:
        text = 'Result type must contain "values" and "confidences" as keys.'
    if not type(variable["values"]) is list or not type(variable["confidences"]):
        text = 'Results["values"] and Results["confidences"] must respectively be a list.'

    if text != "":
        logger.info(text)
        raise Exception(text)


def write_annotation_to_db(request_form, results: dict, logger):
    global database, scheme, session, annotator, roles
    check_format(results, logger)

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

    if request_form["schemeType"] == "DISCRETE":
        write_discrete_to_db(request_form, results, db_handler, logger)
    elif request_form["schemeType"] == "FREE":
        write_freeform_to_db(request_form, results, db_handler, logger)
    elif request_form["schemeType"] == "CONTINUOUS":
        write_continuous_to_db(request_form, results, db_handler, logger)
    elif request_form["schemeType"] == "POINT":
        pass  # todo
    elif request_form["schemeType"] == "DISCRETE_POLYGON" or request_form["schemeType"] == "POLYGON":
        write_polygons_to_db(request_form, results, db_handler, logger)


def write_freeform_to_db(request_form, results: dict, db_handler, logger):
    '''
    Args:
        request_form ():
        results (dict): {
            'from_to' : {
                <role>.<stream>: {
                    'name': <text>, 'conf': <confidence>
                }
            }
        }
        db_handler ():
        logger ():

    Returns:

    '''
    annotations = {}
    """Temp fix"""
    results.pop('values', None)
    results.pop('confidences', None)

    for frame, results in results.items():
        frame_info = frame.split('_')
        frame_from = float(frame_info[-2])
        frame_to = float(frame_info[-1])
        for stream_id, anno in results.items():

            conf = anno['conf']
            name = anno['name']

            if stream_id not in annotations.keys():
                annotations[stream_id] = []

            annotations[stream_id].append({
                "from": frame_from,
                "to": frame_to,
                "conf": conf,
                "name": name
            })

    for anno_id, anno in annotations.items():
        # TODO does not work with flattened roles
        role, stream = anno_id.split('.')

        db_handler.set_annos(
            database=database,
            scheme=scheme,
            session=session,
            annotator=annotator,
            role=role,
            annos=anno,
        )


def write_discrete_to_db(request_form, results: dict, db_handler, logger):
    # TODO: We only take one role into account in this case. Fix
    #role = roles.split(';')[0]
    role = results['roles']
    frame_size = nova_data_utils.parse_time_string_to_ms(request_form['frameSize'])
    mongo_scheme = db_handler.get_mongo_scheme(scheme, database)
    annos = []
    start_frame = 0

    if request_form['startTime'] != '0':
        annos_db = db_handler.get_annos(dataset=database, scheme=scheme, session=session, annotator=annotator,
                                        roles=role)

        start_frame = float(request_form['startTime']) / 1000
        annos = list(filter(lambda x: float(x['from']) < start_frame, annos_db))

    last_label = None
    current_label_start = 1

    values = results["values"]

    for id, value in enumerate(values):
        # current label is different from the one before
        if not value == last_label and last_label is not None:
            frame_from = str(start_frame + ((current_label_start * frame_size) / 1000.0))
            frame_to = str(start_frame + ((id * frame_size) / 1000.0))
            if last_label < len(mongo_scheme[0]['labels']):
                annos.append({
                    'from': frame_from,
                    'to': frame_to,
                    'conf': results["confidences"][id],
                    'id': int(last_label)
                })
            current_label_start = id
        last_label = value

    db_handler.set_annos(
        database=database,
        scheme=scheme,
        session=session,
        annotator=annotator,
        role=role,
        annos=annos,
    )


def write_continuous_to_db(request_form, results: dict, db_handler, logger):
    role = results['roles']
    #role = roles.split(';')[0]
    annos = []

    if request_form['startTime'] != '0':
        annos_db = db_handler.get_annos(dataset=database, scheme=scheme, session=session, annotator=annotator,
                                        roles=roles)

        start_frame = int(int(request_form['startTime']) / int(request_form['frameSize'][:-2]) + 1)
        annos = annos_db[:start_frame]

    #normalized = (results["values"] - min(results["values"])) / (max(results["values"]) - min(results["values"]))

    for value, confidence in zip(results["values"], results["confidences"]):
        annos.append({
            'score': value,
            'conf': confidence
        })

    db_handler.set_annos(
        database=database,
        scheme=scheme,
        session=session,
        annotator=annotator,
        role=role,
        annos=annos,
    )


def write_polygons_to_db(request_form, results: dict, db_handler, logger):
    mongo_scheme = db_handler.get_mongo_scheme(scheme, database)
    mongo_annotator = db_handler.get_mongo_annotator(annotator, database)
    mongo_role = db_handler.get_mongo_role(roles, database)
    mongo_session = db_handler.get_mongo_session(session, database)

    mongo_annotations = db_handler.get_annotation_docs(
        mongo_scheme,
        mongo_session,
        mongo_annotator,
        mongo_role,
        database,
        db_handler.ANNOTATION_COLLECTION,
    )

    start_frame = int(float(results["start_frame"])) - 1

    mongo_data_id = None
    data_backup_id = None
    if mongo_annotations:
        if mongo_annotations[0]["isLocked"]:
            warnings.warn(f"Can't overwrite locked annotation {str(mongo_annotations[0]['_id'])}")
            return ""
        else:
            mongo_data_id = mongo_annotations[0]["data_id"]
            data_backup_id = mongo_annotations[0]["data_backup_id"]
    logger.info("...fetch documents...")
    # 1. Get the doc (will get merged if other docs are there are nextEntry ID's)
    main_docs = db_handler.get_data_docs_by_prop(mongo_data_id, "_id", database)
    backup_doc = db_handler.get_docs_by_prop(mongo_data_id, "_id", database, db_handler.ANNOTATION_DATA_COLLECTION)[0]
    logger.info("...fill documents...")
    # 2. Fill the doc with the Predictions
    main_docs = update_polygon_doc(main_docs, results["values"], results["confidences"], start_frame)

    # 3. Check if the doc is too large (if, separate it)
    if len(bson.BSON.encode(main_docs)) >= MAX_MONGO_DB_DOC_SIZE:
        main_docs = separate_doc(main_docs)
    if not isinstance(main_docs, list):
        main_docs = [main_docs]
    main_docs[0]['_id'] = mongo_data_id

    logger.info("...separate docs if necessary...")
    # 4 Delete old back-up (with tail-docs)
    db_handler.delete_doc_with_tail(data_backup_id, database)

    logger.info("...upload docs...")
    # 5. Update the backup ID
    backup_doc['_id'] = data_backup_id
    db_handler.insert_doc_by_prop(backup_doc, database, db_handler.ANNOTATION_DATA_COLLECTION)
    db_handler.delete_doc_by_prop(mongo_data_id, "_id", database, db_handler.ANNOTATION_DATA_COLLECTION)

    # 6. Upload the doc(s)
    for doc in main_docs:
        db_handler.insert_doc_by_prop(doc, database, db_handler.ANNOTATION_DATA_COLLECTION)


def separate_doc(doc) -> list:
    left = copy.deepcopy(doc)
    left["_id"] = bson.objectid.ObjectId()
    right = copy.deepcopy(doc)
    right["_id"] = bson.objectid.ObjectId()
    half_count = int(len(doc['labels']) / 2)
    left['labels'] = left['labels'][0:half_count]
    right['labels'] = right['labels'][half_count:]

    result_list = []
    if len(bson.BSON.encode(left)) >= MAX_MONGO_DB_DOC_SIZE:
        result_list += separate_doc(left)
    else:
        result_list.append(left)

    if len(bson.BSON.encode(right)) >= MAX_MONGO_DB_DOC_SIZE:
        result_list += separate_doc(right)
    else:
        result_list.append(right)

    return result_list


def update_polygon_doc(data_doc, polygons, confidences, start_frame):
    current_frame = -1
    for frame_id, frame in enumerate(data_doc['labels']):
        if frame_id >= start_frame + len(polygons):
            return data_doc
        if frame_id >= start_frame:
            polygons_per_frame = polygons[frame_id - start_frame]
            for label_id, polygons_per_label_type in enumerate(polygons_per_frame):
                for polygon_id, polygon in enumerate(polygons_per_label_type):
                    label = label_id + 1
                    points_for_db = []
                    points = np.reshape(polygon, newshape=[int(polygon.shape[0] / 2), 2])
                    for point in points:
                        points_for_db.append({'x': int(point[0]), 'y': int(point[1])})

                    # delete the content of the current frame, the new prediction values have to be set
                    if current_frame != frame_id:
                        current_frame = frame_id
                        data_doc['labels'][frame_id]['polygons'] = []

                    data_doc['labels'][frame_id]['polygons'].append(
                        {'label': label,
                         'confidence': round(confidences[frame_id - start_frame][label_id][polygon_id], 2),
                         'points': points_for_db})

    return data_doc
