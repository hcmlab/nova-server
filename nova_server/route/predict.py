import glob
import importlib
import itertools
import os
import threading

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify
from nova_server.utils import thread_utils, status_utils, log_utils, tfds_utils

predict = Blueprint("predict", __name__)


@predict.route("/predict", methods=["POST"])
def predict_thread():
    if request.method == "POST":
        thread = predict_model(request.form)
        thread_id = thread.name
        status_utils.add_new_job(thread_id, predict.name)
        data = {"job_id": thread_id}
        thread.start()
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def predict_model(request_form):
    def update_progress(msg):
        status_utils.update_progress(threading.current_thread().name, msg)

    # Init logging
    status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.RUNNING)

    update_progress('Initalizing')
    logger = log_utils.get_logger_for_thread(__name__)
    log_conform_request = dict(request_form)
    log_conform_request['password'] = '---'
    logger.info(f"Start predicting with request {log_conform_request}")

    trainer_file = request_form.get("trainerScript")
    if trainer_file is None:
        logger.error('No trainer file has been provided. Exiting.')
        status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.ERROR)
        return

    logger.info(f"Loading trainer script {trainer_file}...")
    spec = importlib.util.spec_from_file_location("trainer", trainer_file)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    logger.info("... done")

    # Building the dataset
    update_progress('Dataloading')
    logger.info("Building dataset from request form...")

    try:
        ds_iter = tfds_utils.dataset_from_request_form(request_form)
    except ValueError as ve:
        status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.ERROR)
        logger.error(f'Error while loading dataset. Exiting.\n{str(ve)}')
        return
    logger.info("... done")

    # extract some basic parameters
    sample_rate = 1000 / ds_iter.frame_size_ms
    annotator = ds_iter.annotator
    roles = ds_iter.roles
    sessions = ds_iter.sessions
    scheme = request_form.get('scheme')
    stream_name = request_form.get("stream").split(" ")[0]

    # Preprocess data
    update_progress('Preprocessing')
    logger.info("Start preprocessing...")
    data_list = list(ds_iter)
    data_list.sort(key=lambda x: int(x["frame"].split("_")[0]))
    logger.info("...done")

    # Load model
    logger.info("Load model...")
    model_path = request_form.get("trainerPath")
    model_path = glob.glob(glob.escape(model_path) + '.model')[0]  # get the complete file path
    model = trainer.load(model_path)
    scheme_type = model_path.split('nova')[1].split(os.sep)[4]  # get scheme type from path
    logger.info("...done")

    # predict on data
    update_progress('Predicting')
    logger.info("Start prediction...")

    results = {s: {} for s in sessions}  # initialise empty result dict
    for session, role in itertools.product(sessions, roles):
        filtered_data = [x for x in data_list if session in x['frame'] and role in x['frame']]  # get specific data
        x = [v[stream_name] for v in filtered_data]
        x_np = np.ma.concatenate(x, axis=0)  # prepare data for inference
        predictions = trainer.predict(model, x_np)

        # create and populate a result dataframe for easier handling
        result_df = pd.DataFrame(filtered_data)
        result_df[scheme] = predictions
        result_df[['start', 'end']] = result_df['frame'].str.split('_', expand=True).iloc[:, -3:-1]
        result_df = result_df.drop(columns=[stream_name, 'frame'])[['start', 'end', scheme]]  # remove unused columns
        result_df['confidence'] = 1
        results[session][role] = result_df

    logger.info("...done")

    # Write annotations back to files
    update_progress('Writeback')
    logger.info("Writing annotations...")

    for session, role in itertools.product(sessions, roles):
        session_path = os.path.join(request_form.get('dataPath'), request_form.get('database'), session)
        session_data = results[session][role]

        # set file paths
        data_file = os.path.join(session_path, f'{role}.{scheme}.{annotator}.test.annotation~')
        header_file = os.path.join(session_path, f'{role}.{scheme}.{annotator}.test.annotation')

        if scheme_type == 'continuous':
            session_data = session_data.drop(columns=['start', 'end'])  # remove timestamps as they are not needed
            header = f'<?xml version="1.0" ?>\n' \
                     f'<annotation ssi-v="3">\n' \
                     f'\t<info ftype="ASCII" size="{len(session_data)}"/>\n' \
                     f'\t<meta annotator="{annotator}" role="{role}"/>\n' \
                     f'\t<scheme name="{scheme}" type="CONTINUOUS" sr="{int(sample_rate)}" min="{session_data[scheme].min()}" max="{session_data[scheme].max()}" />\n' \
                     f'</annotation>'

        elif scheme_type == 'discrete':
            # change datatype of predictions to int
            session_data[scheme] = session_data[scheme].astype(int)

            # load classes from template path
            template_path = request_form.get("templatePath")
            template = ET.parse(template_path)
            trainer_classes = [item.get('name') for item in template.find('classes').findall('item')]
            items = ''.join([f'\t\t<item name="{item}" id="{i}"/>\n' for i, item in enumerate(trainer_classes)])

            header = f'<?xml version="1.0" ?>\n' \
                     f'<annotation ssi-v="3">\n' \
                     f'\t<info ftype="ASCII" size="{len(session_data)}"/>\n' \
                     f'\t<meta annotator="{annotator}" role="{role}"/>\n' \
                     f'\t<scheme name="{scheme}" type="DISCRETE">\n' \
                     f'{items}' \
                     f'\t</scheme>\n' \
                     f'</annotation>'

        elif scheme_type == 'free':
            header = f'<?xml version="1.0" ?>\n' \
                     f'<annotation ssi-v="3">\n' \
                     f'\t<info ftype="ASCII" size="{len(session_data)}"/>\n' \
                     f'\t<meta annotator="{annotator}" role="{role}"/>\n' \
                     f'\t<scheme name="{scheme}" type="FREE" />\n' \
                     f'</annotation>'
        else:
            logger.error('Annotation scheme not implemented. Exiting.')
            status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.ERROR)
            return

        # write the actual files back to disk
        session_data.to_csv(data_file, encoding='UTF8', sep=';', header=False, index=False, float_format='%.2f')

        with open(header_file, 'w', encoding='UTF8') as fi:
            fi.write(header)

    update_progress('...done')

    update_progress('Done')
    status_utils.update_status(threading.current_thread().name, status_utils.JobStatus.FINISHED)
