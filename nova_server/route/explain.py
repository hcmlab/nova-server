from pathlib import Path
from flask import Blueprint, request, jsonify
from nova_server.utils.thread_utils import THREADS
from nova_server.utils.job_utils import get_job_id_from_request_form
from nova_server.utils import thread_utils, job_utils
from nova_server.utils import log_utils
from nova_server.utils import env
from nova_server.exec.execution_handler import NovaExplainHandler
import os
import tensorflow as tf
import io
import ast
import numpy as np
import pandas as pd
import io as inputoutput
import base64
import json
import pickle
import warnings
import site as s
from PIL import Image
from PIL import Image as pilimage
from flask import Blueprint, Response, request
from lime.lime_image import LimeImageExplainer
from lime.lime_tabular import LimeTabularExplainer
import tf_explain
import dice_ml



def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize the input image and preprocess it
    image = image.resize(target)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # return the processed image
    return image

def getTopXpredictions(prediction, topLabels):

    prediction_class = []

    for i in range(0, len(prediction[0])):
        prediction_class.append((i, prediction[0][i]))

    prediction_class.sort(key=lambda x: x[1], reverse=True)

    return prediction_class[:topLabels]


explain = Blueprint("explain", __name__)


@explain.route("/explain", methods=["POST"])
def explain_thread():
        if request.method == "POST":
            request_form = request.form.to_dict()
            job_id = get_job_id_from_request_form(request_form)
            job_added = job_utils.add_new_job(job_id, request_form=request_form)
            thread = explain_data(request_form, job_id)
            thread.start()
            THREADS[job_id] = thread
            data = {"success": str(job_added)}
            return jsonify(data)

            
            explanation_framework = request_form.get("explainer")

            if explanation_framework == "LIME_IMAGE":
                return lime_image(request_form)
            elif explanation_framework == "LIME_TABULAR":
                return lime_tabular(request_form)
            elif explanation_framework == "DICE":
                return dice(request_form)
            elif explanation_framework == "TF_EXPLAIN":
                return tf_explainer(request_form)
            return None


@thread_utils.ml_thread_wrapper
def explain_data(request_form, job_id):
    #job_id = get_job_id_from_request_form(request_form)
    job_utils.update_status(job_id, job_utils.JobStatus.RUNNING)
    logger = log_utils.get_logger_for_job(job_id)
    logger.info(request_form)

    job = job_utils.get_job(job_id)
    job.execution_handler = NovaExplainHandler(request_form, logger=logger, backend=os.getenv(env.NOVA_SERVER_BACKEND))

    try:
        job.run()
        job_utils.update_status(job_id, job_utils.JobStatus.FINISHED)
    except Exception as e:
        logger.critical(f"Job failed with exception {str(e)}")
        job_utils.update_status(job_id, job_utils.JobStatus.ERROR)

def tf_explainer(request):
    data = {"success": "failed"}

    frame = request.get("frame")
    explainer = request.get("tfExplainer")

    nova_iterator = NovaIterator(
        request.get("dbHost"),
        int(request.get("dbPort")),
        request.get("dbUser"),
        request.get("dbPassword"),
        request.get("dataset"),
        os.environ["NOVA_SERVER_DATA_DIR"],
        ast.literal_eval(request.get("sessions")),
        ast.literal_eval(request.get("data")),
        0
        )

    # NOTE: loading model before initializing data results in winerror insufficient system resources
    session = nova_iterator.init_session(ast.literal_eval(request.get("sessions"))[0])
    sample = session.input_data["explanation_stream"].data[frame]

    model_path = request.get("modelPath")
    model = tf.keras.models.load_model(model_path)


    image = prepare_image(Image.fromarray(sample), (224, 224))
    image = image * (1.0 / 255)
    prediction = model.predict(image)
    topClass = getTopXpredictions(prediction, 1)
    print(topClass[0])
    image = np.squeeze(image)

    if explainer == "GRADCAM":
        im = ([image], None)
        from tf_explain.core.grad_cam import GradCAM

        exp = GradCAM()
        imgFinal = exp.explain(im, model, class_index=topClass[0][0])
        # exp.save(imgFinal, ".", "grad_cam.png")

    elif explainer == "OCCLUSIONSENSITIVITY":
        im = ([image], None)
        from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity

        exp = OcclusionSensitivity()
        imgFinal = exp.explain(
            im, model, class_index=topClass[0][0], patch_size=10
        )
        # exp.save(imgFinal, ".", "grad_cam.png")

    elif explainer == "GRADIENTSINPUTS":
        im = (np.array([image]), None)
        from tf_explain.core.gradients_inputs import GradientsInputs

        exp = GradientsInputs()
        imgFinal = exp.explain(im, model, class_index=topClass[0][0])
        # exp.save(imgFinal, ".", "gradients_inputs.png")

    elif explainer == "VANILLAGRADIENTS":
        im = (np.array([image]), None)
        from tf_explain.core.vanilla_gradients import VanillaGradients

        exp = VanillaGradients()
        imgFinal = exp.explain(im, model, class_index=topClass[0][0])
        # exp.save(imgFinal, ".", "gradients_inputs.png")

    elif explainer == "SMOOTHGRAD":
        im = (np.array([image]), None)
        from tf_explain.core.smoothgrad import SmoothGrad

        exp = SmoothGrad()
        imgFinal = exp.explain(im, model, class_index=topClass[0][0])
        # exp.save(imgFinal, ".", "gradients_inputs.png")

    elif explainer == "INTEGRATEDGRADIENTS":
        im = (np.array([image]), None)
        from tf_explain.core.integrated_gradients import IntegratedGradients

        exp = IntegratedGradients()
        imgFinal = exp.explain(im, model, class_index=topClass[0][0])
        # exp.save(imgFinal, ".", "gradients_inputs.png")

    elif explainer == "ACTIVATIONVISUALIZATION":
        # need some solution to find out and submit layers name
        im = (np.array([image]), None)
        from tf_explain.core.activations import ExtractActivations

        exp = ExtractActivations()
        imgFinal = exp.explain(im, model, layers_name=["activation_1"])
        # exp.save(imgFinal, ".", "gradients_inputs.png")

    img = pilimage.fromarray(imgFinal)
    imgByteArr = inputoutput.BytesIO()
    img.save(imgByteArr, format="JPEG")
    imgByteArr = imgByteArr.getvalue()

    img64 = base64.b64encode(imgByteArr)
    img64_string = img64.decode("utf-8")

    data["explanation"] = img64_string
    data["prediction"] = str(topClass[0][0])
    data["prediction_score"] = str(topClass[0][1])
    data["success"] = "success"

    return json.dumps(data)
