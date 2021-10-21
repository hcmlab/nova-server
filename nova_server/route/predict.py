import io
from flask import Blueprint, request, jsonify
from PIL import Image
from nova_server.utils import img_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import imagenet_utils


predict = Blueprint("predict", __name__)

@predict.route("/predict", methods=["POST"])
def predict():

    global graph
    data = {"success": "failed"}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.form.get("image"):
            # read the image in PIL format
            image = request.form.get("image")
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = img_utils.prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            with graph.as_default():
                model_path = request.form.get("model_path")
                model = load_model(model_path)

                preds = model.predict(image)
                results = imagenet_utils.decode_predictions(preds)
                data["predictions"] = []

                # loop over the results and add them to the list of
                # returned predictions
                for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

                # indicate that the request was a success
                data["success"] = True

    # return the data dictionary as a JSON response
    return jsonify(data)
