import sys
import bson
import copy
import warnings
import numpy as np

from hcai_datasets.hcai_nova_dynamic.nova_db_handler import NovaDBHandler
from src.ssi_utils.ssi_anno_utils import Anno, SchemeType

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
        text = "Result type must be a dictionary!"
    if "values" not in variable or "confidences" not in variable:
        text = 'Result type must contain "values" and "confidences" as keys.'
    if not type(variable["values"]) is list or not type(variable["confidences"]):
        text = (
            'Results["values"] and Results["confidences"] must respectively be a list.'
        )

    if text != "":
        logger.info(text)
        raise Exception(text)


def write_stream_info_to_db(
    request_form: dict,
    file_name: str,
    file_ext: str,
    stream_type: str,
    is_valid: bool,
    sr: float,
    dim_labels: list = None,
):
    # TODO check if we really need to establish a new connection to the database
    db_config_dict = {
        "ip": request_form["dbServer"].split(":")[0],
        "port": int(request_form["dbServer"].split(":")[1]),
        "user": request_form["dbUser"],
        "password": request_form["dbPassword"],
    }

    db_handler = NovaDBHandler(db_config_dict=db_config_dict)
    database = request_form["database"]
    if file_ext.startswith("."):
        file_ext = file_ext[1:]
    db_handler.set_data_streams(
        database=database,
        file_name=file_name,
        file_ext=file_ext,
        stream_type=stream_type,
        is_valid=is_valid,
        sr=sr,
        dimlabels=dim_labels,
        overwrite=True,
    )


def write_annotation_to_db(request_form, anno: Anno, logger):
    # global database, scheme, session, annotator, roles
    # check_format(results, logger)

    # TODO check if we really need to establish a new connection to the database
    # DB Config
    db_config_dict = {
        "ip": request_form["dbServer"].split(":")[0],
        "port": int(request_form["dbServer"].split(":")[1]),
        "user": request_form["dbUser"],
        "password": request_form["dbPassword"],
    }

    # Database specific options
    db_handler = NovaDBHandler(db_config_dict=db_config_dict)
    database = request_form["database"]
    session = request_form["sessions"]

    # Format data correctly
    scheme_dtype_names = anno.scheme.get_dtype().base.names
    anno_data = [dict(zip(scheme_dtype_names, ad.item())) for ad in anno.data]

    db_handler.set_annos(
        database=database,
        session=session,
        scheme=anno.scheme.name,
        annotator=anno.annotator,
        role=anno.role,
        annos=anno_data,
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
            warnings.warn(
                f"Can't overwrite locked annotation {str(mongo_annotations[0]['_id'])}"
            )
            return ""
        else:
            mongo_data_id = mongo_annotations[0]["data_id"]
            data_backup_id = mongo_annotations[0]["data_backup_id"]
    logger.info("...fetch documents...")
    # 1. Get the doc (will get merged if other docs are there are nextEntry ID's)
    main_docs = db_handler.get_data_docs_by_prop(mongo_data_id, "_id", database)
    backup_doc = db_handler.get_docs_by_prop(
        mongo_data_id, "_id", database, db_handler.ANNOTATION_DATA_COLLECTION
    )[0]
    logger.info("...fill documents...")
    # 2. Fill the doc with the Predictions
    main_docs = update_polygon_doc(
        main_docs, results["values"], results["confidences"], start_frame
    )

    # 3. Check if the doc is too large (if, separate it)
    if len(bson.BSON.encode(main_docs)) >= MAX_MONGO_DB_DOC_SIZE:
        main_docs = separate_doc(main_docs)
    if not isinstance(main_docs, list):
        main_docs = [main_docs]
    main_docs[0]["_id"] = mongo_data_id

    logger.info("...separate docs if necessary...")
    # 4 Delete old back-up (with tail-docs)
    db_handler.delete_doc_with_tail(data_backup_id, database)

    logger.info("...upload docs...")
    # 5. Update the backup ID
    backup_doc["_id"] = data_backup_id
    db_handler.insert_doc_by_prop(
        backup_doc, database, db_handler.ANNOTATION_DATA_COLLECTION
    )
    db_handler.delete_doc_by_prop(
        mongo_data_id, "_id", database, db_handler.ANNOTATION_DATA_COLLECTION
    )

    # 6. Upload the doc(s)
    for doc in main_docs:
        db_handler.insert_doc_by_prop(
            doc, database, db_handler.ANNOTATION_DATA_COLLECTION
        )


# def separate_doc(doc) -> list:
#     left = copy.deepcopy(doc)
#     left["_id"] = bson.objectid.ObjectId()
#     right = copy.deepcopy(doc)
#     right["_id"] = bson.objectid.ObjectId()
#     half_count = int(len(doc["labels"]) / 2)
#     left["labels"] = left["labels"][0:half_count]
#     right["labels"] = right["labels"][half_count:]
#
#     result_list = []
#     if len(bson.BSON.encode(left)) >= MAX_MONGO_DB_DOC_SIZE:
#         result_list += separate_doc(left)
#     else:
#         result_list.append(left)
#
#     if len(bson.BSON.encode(right)) >= MAX_MONGO_DB_DOC_SIZE:
#         result_list += separate_doc(right)
#     else:
#         result_list.append(right)
#
#     return result_list
#
#
# def update_polygon_doc(data_doc, polygons, confidences, start_frame):
#     current_frame = -1
#     for frame_id, frame in enumerate(data_doc["labels"]):
#         if frame_id >= start_frame + len(polygons):
#             return data_doc
#         if frame_id >= start_frame:
#             polygons_per_frame = polygons[frame_id - start_frame]
#             for label_id, polygons_per_label_type in enumerate(polygons_per_frame):
#                 for polygon_id, polygon in enumerate(polygons_per_label_type):
#                     label = label_id + 1
#                     points_for_db = []
#                     points = np.reshape(
#                         polygon, newshape=[int(polygon.shape[0] / 2), 2]
#                     )
#                     for point in points:
#                         points_for_db.append({"x": int(point[0]), "y": int(point[1])})
#
#                     # delete the content of the current frame, the new prediction values have to be set
#                     if current_frame != frame_id:
#                         current_frame = frame_id
#                         data_doc["labels"][frame_id]["polygons"] = []
#
#                     data_doc["labels"][frame_id]["polygons"].append(
#                         {
#                             "label": label,
#                             "confidence": round(
#                                 confidences[frame_id - start_frame][label_id][
#                                     polygon_id
#                                 ],
#                                 2,
#                             ),
#                             "points": points_for_db,
#                         }
#                     )
#
#     return data_doc
