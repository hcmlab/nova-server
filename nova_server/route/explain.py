from pathlib import Path
from flask import Blueprint, request, jsonify
from nova_server.utils.thread_utils import THREADS
from nova_server.utils.job_utils import get_job_id_from_request_form
from nova_server.utils import thread_utils, job_utils
from nova_server.utils import log_utils
from nova_server.exec.execution_handler import NovaExtractHandler
from nova_server.exec.execution_handler import NovaProcessHandler
import os
import tensorflow as tf
import io
import ast
import numpy as np
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
from skimage.segmentation import mark_boundaries
from nova_utils.data.handler.mongo_handler import (
    AnnotationHandler,
    StreamHandler,
    SessionHandler,
)
from nova_utils.data.provider.nova_iterator import NovaIterator

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
            key = get_job_id_from_request_form(request_form)
            job_utils.add_new_job(key, request_form=request_form)
            # thread = process_data(request_form)
            # thread.start()
            # THREADS[key] = thread
            # data = {"success": "true"}
            # return jsonify(data)
            
            explanation_framework = request_form.get("explainer")

            if explanation_framework == "LIME_IMAGE":
                return lime_image(request_form)
            elif explanation_framework == "LIME_TABULAR":
                return lime_tabular(request_form)
            elif explanation_framework == "TF_EXPLAIN":
                return tf_explainer(request_form)
            return None


@thread_utils.ml_thread_wrapper
def process_data(request_form):
    job_id = get_job_id_from_request_form(request_form)

    job_utils.update_status(job_id, job_utils.JobStatus.RUNNING)
    logger = log_utils.get_logger_for_job(job_id)
    logger.info(request_form)

def lime_image(request):
    data = {"success": "failed"}

    frame = int(request.get("frame"))
    num_features = int(request.get("numFeatures"))
    top_labels = int(request.get("topLabels"))
    num_samples = int(request.get("numSamples"))
    hide_color = request.get("hideColor")
    hide_rest = request.get("hideRest")
    positive_only = request.get("positiveOnly")

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

    img = prepare_image(Image.fromarray(sample), (224, 224))
    img = img * (1.0 / 255)
    prediction = model.predict(img)
    explainer = LimeImageExplainer()
    img = np.squeeze(img).astype("double")
    explanation = explainer.explain_instance(
        img,
        model.predict,
        top_labels=top_labels,
        hide_color=hide_color == "True",
        num_samples=num_samples,
    )

    top_classes = getTopXpredictions(prediction, top_labels)

    explanations = []

    for cl in top_classes:
        temp, mask = explanation.get_image_and_mask(
            cl[0],
            positive_only=positive_only == "True",
            num_features=num_features,
            hide_rest=hide_rest == "True",
        )
        img_explained = mark_boundaries(temp, mask)
        img = Image.fromarray(np.uint8(img_explained * 255))
        img_byteArr = io.BytesIO()
        img.save(img_byteArr, format="JPEG")
        img_byteArr = img_byteArr.getvalue()
        img64 = base64.b64encode(img_byteArr)
        img64_string = img64.decode("utf-8")

        explanations.append((str(cl[0]), str(cl[1]), img64_string))

    data["explanations"] = explanations
    data["success"] = "success"

    return json.dumps(data)


def lime_tabular(request):
    data = {"success": "failed"}

    with open(request.get("modelPath"), "rb") as f:
        model = pickle.load(f)

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

    stream_data = list(nova_iterator)[0]["explanation_stream"]

    frame = int(request.get("frame"))
    sample = stream_data[frame]
    top_class = getTopXpredictions(model.predict_proba([sample]), 1)
    num_features = int(request.get("numFeatures"))
    explainer = LimeTabularExplainer(
        stream_data, mode="classification", discretize_continuous=True
    )
    exp = explainer.explain_instance(
        np.asarray(sample),
        model.predict_proba,
        num_features=num_features,
        top_labels=1,
    )

    explanation_dictionary = {}

    for entry in exp.as_list(top_class[0][0]):
        explanation_dictionary.update({entry[0]: entry[1]})


    data = {"success": "true",
            "explanation": explanation_dictionary}
    return jsonify(data)

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