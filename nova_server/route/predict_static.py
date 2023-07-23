"""This file contains the general logic for predicting annotations to the nova database"""
import copy
import json
import os
from pathlib import Path, PureWindowsPath
from nova_server.utils import db_utils
from flask import Blueprint, request, jsonify
from nova_utils.ssi_utils.ssi_xml_utils import Trainer
from importlib.machinery import SourceFileLoader
from nova_server.utils.thread_utils import THREADS
from nova_server.utils.status_utils import update_progress
from nova_server.utils.key_utils import get_key_from_request_form
from nova_server.utils import (
    thread_utils,
    status_utils,
    log_utils,
    dataset_utils,
    import_utils,
    nostr_utils
)
from hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable import (
    HcaiNovaDynamicIterable,
)
from nova_utils.interfaces.server_module import Trainer as iTrainer

predict_static = Blueprint("predict_static", __name__)


@predict_static.route("/predict_static", methods=["POST"])
def predict_static_thread():
    if request.method == "POST":
        request_form = request.form.to_dict()
        key = get_key_from_request_form(request_form)
        request_form['jobID'] = key
        thread = predict_static_data(request_form)
        status_utils.add_new_job(key, request_form=request_form)
        data = {"success": "true"}
        thread.start()
        THREADS[key] = thread
        return jsonify(data)


@thread_utils.ml_thread_wrapper
def predict_static_data(request_form):
    key = get_key_from_request_form(request_form)
    logger = log_utils.get_logger_for_thread(key)

    log_conform_request = dict(request_form)

    logger.info("Action 'Predict Static' started.")
    status_utils.update_status(key, status_utils.JobStatus.RUNNING)

    update_progress(key, "Data loading")

    task = request_form['trainerFilePath']
    logger.info("Setting options...")
    options = []
    if request_form.get("optStr"):
        for k, v in [
            option.split("=") for option in request_form["optStr"].split(";")]:
            t = (k, v)
            options.append(v)
            logger.info(k + "=" + v)
    logger.info("...done.")

    if task == "text-to-image":
        anno = textToImage(options[0], options[1])
    elif task == "image-to-image":
        anno = imageToImage(options[0], options[1], options[2], options[3], options[4])
    elif task == "image-upscale":
        anno = imageUpscale(options[0])
    elif task == "translation":
        anno = GoogleTranslate(options[0], options[1])

    if request_form["nostrEvent"] is not None:
        nostr_utils.CheckEventStatus(anno, str(request_form["nostrEvent"]), request_form["isBot"])
    logger.info("...done")

    logger.info("Prediction completed!")
    status_utils.update_status(key, status_utils.JobStatus.FINISHED)


# HELPER
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "(" + str(counter) + ")" + extension
        counter += 1

    return path


def uploadToHoster(filepath):
    import requests
    try:
        files = {'file': open(filepath, 'rb')}
        url = 'https://nostrfiles.dev/upload_image'
        response = requests.post(url, files=files)
        json_object = json.loads(response.text)
        print(json_object["url"])
        return json_object["url"]
    except:
        # fallback filehoster
        files = {'image': open(filepath, 'rb')}
        url = 'https://nostr.build/api/upload/android.php'
        response = requests.post(url, files=files)
        result = response.text.replace("\\", "").replace("\"", "")
        print(result)
        return result


# SCRIPTS (TO BE MOVED TO FILES)
def textToImage(prompt, negative_prompt):
    import torch
    from diffusers import DiffusionPipeline
    from diffusers import StableDiffusionPipeline

    # model_id_or_path = "runwayml/stable-diffusion-v1-5"
    # pipe = DiffusionPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)

    pipe = StableDiffusionPipeline.from_single_file(
        "sdmodels/stablydiffusedsWild_351.safetensors"
    )
    # pipe.unet.load_attn_procs(model_id_or_path)
    pipe = pipe.to("cuda")
    image = pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
    uniquefilepath = uniquify("outputs/sd.jpg")
    image.save(uniquefilepath)
    return uploadToHoster(uniquefilepath)

def imageToImage(url, prompt, negative_prompt, strength, guidance_scale):
    import requests
    import torch
    from PIL import Image
    from io import BytesIO

    from diffusers import StableDiffusionImg2ImgPipeline

    device = "cuda"
    # model_id_or_path = "runwayml/stable-diffusion-v1-5"
    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)

    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        "sdmodels/stablydiffusedsWild_351.safetensors"
    )
    # pipe.unet.load_attn_procs(model_id_or_path)
    pipe = pipe.to(device)

    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    # init_image = init_image.resize((768, 512))

    image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, strength=strength,
                 guidance_scale=guidance_scale).images[0]
    uniquefilepath = uniquify("outputs/sd.jpg")
    return uploadToHoster((uniquefilepath))


def imageUpscale(url):
    import requests
    from PIL import Image
    from io import BytesIO
    from diffusers import StableDiffusionUpscalePipeline
    import torch

    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")
    pipeline.enable_attention_slicing()

    response = requests.get(url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = low_res_img.resize((256, 256))  # This is bad but memory is too low.

    prompt = "UHD, 4k, hyper realistic, extremely detailed, professional, vibrant, not grainy, smooth, sharp"
    upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
    uniquefilepath = uniquify("outputs/sd.jpg")
    upscaled_image.save(uniquefilepath)
    return uploadToHoster(uniquefilepath)

def GoogleTranslate(text,translation_lang):
    from translatepy.translators.google import GoogleTranslate
    gtranslate = GoogleTranslate()
    try:
        translated_text = str(gtranslate.translate(text, translation_lang))
        print("Translated Text: " + translated_text)
    except:
        translated_text = "An error occured"
    return translated_text

