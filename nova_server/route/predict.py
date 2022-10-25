import importlib.util

from flask import Blueprint, request, jsonify
from nova_server.utils.thread_utils import THREADS
from nova_server.utils.status_utils import update_progress
from nova_server.utils.key_utils import get_key_from_request_form
from nova_server.utils import status_utils, thread_utils, log_utils, tfds_utils
from nova_server.utils.polygon_utils import prediction_to_binary_mask, mask_to_polygons, get_confidence_from_prediction


predict = Blueprint("predict", __name__)


@predict.route("/predict", methods=["POST"])
def predict_thread():
    if request.method == "POST":
        request_form = request.form.to_dict()
        key = get_key_from_request_form(request_form)
        thread = predict_thread_function(request_form)
        status_utils.add_new_job(key)
        data = {"success": "true"}
        thread.start()
        THREADS[key] = thread
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def predict_thread_function(request_form):
    request_form = request_form.to_dict(flat=False)
    predict_data(request_form)


def predict_data(request_form):
    key = get_key_from_request_form(request_form)
    logger = log_utils.get_logger_for_thread(key)
    logger.info("Action 'Predict' started.")

    trainer_file = request_form["trainerScript"]
    if trainer_file is None:
        logger.error("Trainer file not available!")
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return
    else:
        logger.info("Trainer file available...")

    spec = importlib.util.spec_from_file_location("trainer", trainer_file)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    try:
        update_progress(key, 'Dataloading')
        ds_iter = tfds_utils.dataset_from_request_form(request_form)
        logger.info("Prediction-Data successfully loaded...")
    except ValueError:
        log_utils.remove_log_from_dict(key)
        logger.error("Not able to load the data from the database!")
        status_utils.update_status(key, status_utils.JobStatus.ERROR)
        return

    # TODO MARCO: data_list enthält nicht alle Frames...
    data_list = list(ds_iter)

    logger.info("Trying to start predictions...")

    if request_form["schemeType"] == "DISCRETE_POLYGON" or request_form["schemeType"] == "POLYGON":
        data_list.sort(key=lambda x: int(x[request_form["scheme"]]['name']))
        amount_of_labels = len(ds_iter.label_info[list(ds_iter.label_info)[0]].labels) + 1

        model = trainer.load(request_form["weightsPath"], amount_of_labels)
        predictions = trainer.predict(model, data_list, logger)

        binary_masks = prediction_to_binary_mask(predictions, label_count=amount_of_labels)
        all_polygons = mask_to_polygons(binary_masks)
        confidences = get_confidence_from_prediction(predictions, all_polygons)

    elif request_form["schemeType"] == "DISCRETE":
        # TODO Marco
        ...
    elif request_form["schemeType"] == "FREE":
        # TODO Marco
        ...
    elif request_form["schemeType"] == "CONTINUOUS":
        # TODO Marco
        ...
    elif request_form["schemeType"] == "POINT":
        # TODO
        ...

    # TODO MARCO predictions in DB schreiben (in Nova muss danach neu geladen werden)

    status_utils.update_status(key, status_utils.JobStatus.FINISHED)
    print()
    ...
