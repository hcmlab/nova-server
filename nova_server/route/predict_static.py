"""This file contains the general logic for predicting annotations to the nova database"""
import copy
import gc
import json
import os
import re
from datetime import timedelta
import random

import numpy as np
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline, \
    EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image
from flask import Blueprint, request, jsonify
from huggingface_hub import model_info
from nostr_sdk import PublicKey, Timestamp, Tag, EventId

from nova_server.utils.thread_utils import THREADS
from nova_server.utils.status_utils import update_progress
from nova_server.utils.key_utils import get_key_from_request_form, str2bool
from nova_server.utils import (
    thread_utils,
    status_utils,
    log_utils,
    nostr_utils
)
from hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable import (
    HcaiNovaDynamicIterable,
)
from nova_utils.interfaces.server_module import Trainer as iTrainer

os.environ['TRANSFORMERS_CACHE'] = 'W:/nova/cml/models/trainer/.cache/'

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
    opts = []
    if request_form.get("optStr"):
        for k, v in [option.split("=") for option in request_form["optStr"].split(";")]:
            t = (k, v)
            opts.append(t)
            logger.info(k + "=" + v)
    logger.info("...done.")
    options = dict(opts)
    # TODO move these to separate files
    try:
        if task == "text-to-image":
            anno = textToImage(options["prompt"], options["extra_prompt"], options["negative_prompt"], options["width"],
                               options["height"], options["upscale"], options["model"],  options["ratiow"], options["ratioh"], options["lora"])
        elif task == "image-to-image":
            anno = imageToImage(options["url"], options["prompt"], options["negative_prompt"], options["strength"],
                                options["guidance_scale"], options["model"])
        elif task == "image-upscale":
            anno = imageUpscaleRealESRGANUrl(options["url"], options["upscale"])
        elif task == "translation":
            anno = GoogleTranslate(options["text"], options["translation_lang"])
        elif task == "image-to-text":
            anno = OCRtesseract(options["url"])
        elif task == "chat":
            anno = LLAMA2(options["message"], options["user"])
        elif task == "summarization":
            anno = LLAMA2(
                "Give me a summarization of the most important points of the following text: " + options["message"],  options["user"], options["system_prompt"])
        elif task == "inactive-following":
            anno = InactiveNostrFollowers(options["user"], int(options["since"]), str2bool(options["is_bot"]))
        elif task == "note-recommendation":
            anno = NoteRecommendations(options["user"], int(options["since"]), str2bool(options["is_bot"]))


        if "nostrEvent" in request_form:
            if request_form["nostrEvent"] is not None :
                nostr_utils.check_event_status(anno, str(request_form["nostrEvent"]), str2bool(request_form["isBot"]))
        logger.info("...done")

        logger.info("Prediction completed!")
        status_utils.update_status(key, status_utils.JobStatus.FINISHED)
    except Exception  as e:
        nostr_utils.respond_to_error(str(e), str(request_form["nostrEvent"]), str2bool(request_form["isBot"]))




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
        url = 'https://nostr.build/api/v2/upload/files'
        response = requests.post(url, files=files)
        json_object = json.loads(response.text)
        result = json_object["data"][0]["url"]
        print(result)
        return result

    except:
        files = {'file': open(filepath, 'rb')}
        url = 'https://nostrfiles.dev/upload_image'
        response = requests.post(url, files=files)
        json_object = json.loads(response.text)
        print(json_object["url"])
        return json_object["url"]
        # fallback filehoster


# SCRIPTS (TO BE MOVED TO FILES)
def textToImage(prompt, extra_prompt="", negative_prompt="", widthst="512", heightst="512", upscale="1",
                model="stabilityai/stable-diffusion-xl-base-1.0", ratiow="1", ratioh="1", lora=""):
    import torch
    from diffusers import DiffusionPipeline
    from diffusers import StableDiffusionPipeline


    if model.__contains__("realistic"):
        model = "realisticVisionV51_v51VAE"
    elif model.__contains__("sd15"):
        model = "runwayml/stable-diffusion-v1-5"
    elif model.__contains__("sd21"):
        model = "stabilityai/stable-diffusion-2-1"
    elif model.__contains__("wild"):
        model = "stablydiffusedsWild_351"
    elif model.__contains__("dreamshaper"):
        model = "dreamshaper_8"
    elif model.__contains__("lora"):
        model = model
    else:
        model = "stabilityai/stable-diffusion-xl-base-1.0"

    if extra_prompt != "":
        prompt = prompt + "," + extra_prompt

    mwidth = 768
    mheight = 768

    if  model == "dreamshaper_8" or model == "stabilityai/stable-diffusion-xl-base-1.0":
        mwidth = 1024
        mheight = 1024


    height = min(int(heightst), mheight)
    width = min(int(widthst), mwidth)

    ratiown = int(ratiow)
    ratiohn= int(ratioh)

    if ratiown > ratiohn:
        height = int((ratiohn/ratiown) * float(width))
    elif ratiown < ratiohn:
        width =  int((ratiown/ratiohn) * float(height))
    elif ratiown == ratiohn:
        width = height


    if model == "stabilityai/stable-diffusion-xl-base-1.0":

        base = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        base.to("cuda")

        existing_lora = False
        if lora != "":
            loramodelsfolder = os.environ['TRANSFORMERS_CACHE'] + "stablediffusionmodels/lora/"
            if lora == "cyborg_style_xl":
                prompt = " cyborg style, " + prompt + " <lora:cyborg_style_xl:.8>"
                existing_lora = True

            if lora == "3d_render_style_xl":
                prompt = "3d style, 3d render, " + prompt + " <lora:3d_render_style_xl:1>"
                existing_lora = True

            if lora == "psychedelic_noir_xl":
                prompt = prompt + " <lora:Psychedelic_Noir__sdxl:1.0>"
                existing_lora = True

            if lora == "wojak_xl":
                prompt = "<lora:wojak_big:1>, " + prompt + ", wojak"
                existing_lora = True

            if lora == "dreamarts_xl":
                prompt = "<lora:DreamARTSDXL:1>, " + prompt
                existing_lora = True

            if lora == "voxel_xl":
                prompt = "voxel style, " + prompt + " <lora:last:1>"
                existing_lora = True

            if lora == "kru3ger_xl":
                prompt = "kru3ger_style, " + prompt + "<lora:sebastiankrueger-kru3ger_style-000007:1>"
                existing_lora = True


            if existing_lora:
                lora_weights = loramodelsfolder + lora + ".safetensors"
                base.load_lora_weights(lora_weights)


        # base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        #refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
        # Define how many steps and what % of steps to be run on each experts (80/20) here

        n_steps = 35
        high_noise_frac = 0.8
        image = base(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            negative_prompt=negative_prompt,
            output_type="latent",
        ).images

        if torch.cuda.is_available():
            del base
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        refiner.to("cuda")
        # refiner.enable_model_cpu_offload()
        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            negative_prompt=negative_prompt,
            image=image,
        ).images[0]
        if torch.cuda.is_available():
            del  refiner
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    elif model.startswith("lora"):
        split = model.split("_")
        if len(split) > 1:
            model = split[1]
        else:
            model = "inks"
        from diffusers import StableDiffusionPipeline
        import torch

        loramodelsfolder = os.environ['TRANSFORMERS_CACHE'] + "stablediffusionmodels/lora/"
        base_model = os.environ[
                         'TRANSFORMERS_CACHE'] + "stablediffusionmodels/anyloraCheckpoint_bakedvaeBlessedFp16.safetensors"
        base_model = os.environ[
                         'TRANSFORMERS_CACHE'] + "stablediffusionmodels/dreamshaper_8.safetensors"


        if model == "inks":
            # local lora models
            if model == "inks":
                model_path = loramodelsfolder + "ink_scenery.safetensors"
                prompt = "white background, scenery, ink, mountains, water, trees, " + prompt + " <lora:ink-0.1-3-b28-bf16-D128-A1-1-ep64-768-DAdaptation-cosine:1>"




            pipe = StableDiffusionPipeline.from_single_file(base_model, torch_dtype=torch.float16,
                                                            safety_checker = None, requires_safety_checker = False)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.to("cuda")

            pipe.load_lora_weights(model_path)
        else:
            # huggingface repo lora models
            if model == "t4":
                model_path = "sayakpaul/sd-model-finetuned-lora-t4"
            elif model == "pokemon":
                model_path = "pcuenq/pokemon-lora"

            info = model_info(model_path)
            base_model = info.cardData["base_model"]
            pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.to("cuda")
            pipe.unet.load_attn_procs(model_path)

        image = pipe(prompt, num_inference_steps=30,  width=width,
            height=height, guidance_scale=7.5, cross_attention_kwargs={"scale": 1.0}).images[
            0]

        if torch.cuda.is_available():
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    else:
        if model == "runwayml/stable-diffusion-v1-5":
            pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16,
                                                     use_safetensors=True, variant="fp16"
                                                     )
        else:
            pipe = StableDiffusionPipeline.from_single_file(
                os.environ['TRANSFORMERS_CACHE'] + "stablediffusionmodels/" + model + ".safetensors",
                safety_checker = None, requires_safety_checker = False)

        # pipe.unet.load_attn_procs(model_id_or_path)
        pipe = pipe.to("cuda")
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=width,
                     height=height).images[0]
        if torch.cuda.is_available():
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    uniquefilepath = uniquify("outputs/sd.jpg")
    image.save(uniquefilepath)

    if (int(upscale) > 1 and int(upscale) <= 4):
        print("Upscaling by factor " + upscale + " using RealESRGAN")
        uniquefilepath = imageUpscaleRealESRGAN(uniquefilepath, upscale)

    return uploadToHoster(uniquefilepath)


def imageToImage(url, prompt, negative_prompt, strength, guidance_scale, model="pix2pix"):
    import requests
    import torch
    from PIL import Image
    from io import BytesIO

    from diffusers import StableDiffusionImg2ImgPipeline

    mwidth = 768
    mheight = 768


    if model.__contains__("gta"):
        model = "GTA5_Artwork_Diffusion_gtav_style"
    elif model.__contains__("realistic"):
        model = "realisticVisionV51_v51VAE"
    elif model.__contains__("sdxl"):
        model = "stabilityai/stable-diffusion-xl-refiner-1.0"
    elif model.__contains__("wild"):
        model = "stablydiffusedsWild_351"
    elif model.__contains__("pix2pix"):
        mwidth = 768
        mheight = 768
        model = "timbrooks/instruct-pix2pix"
    else:
        model = "timbrooks/instruct-pix2pix"

    init_image = load_image(url).convert("RGB")



    if  model == "dreamshaper_8" or model == "stabilityai/stable-diffusion-xl-refiner-1.0":
        mwidth = 1024
        mheight = 1024



    if init_image.width > init_image.height:
        scale = float(init_image.height / init_image.width)
        init_image = init_image.resize((mwidth,  int(mheight * scale)))
    elif init_image.width < init_image.height:
        scale = float(init_image.width / init_image.height)
        init_image = init_image.resize((int(mwidth * scale), mheight))
    else:
        init_image = init_image.resize((mwidth, mheight))


    if model == "stabilityai/stable-diffusion-xl-refiner-1.0":

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model, torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True
        )
        n_steps = 35
        high_noise_frac = 0.8

        pipe = pipe.to("cuda")
        image = pipe(prompt, image=init_image,
                     negative_prompt=negative_prompt, ).images[0]





    elif model == "timbrooks/instruct-pix2pix":
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model, torch_dtype=torch.float16,
                                                                      safety_checker=None)

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image).images[0]

        # 'CompVis/stable-diffusion-v1-4'
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            os.environ['TRANSFORMERS_CACHE'] + "stablediffusionmodels/" + model + ".safetensors"
        )
        pipe = pipe.to("cuda")

        image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, strength=float(strength),
                     guidance_scale=float(guidance_scale)).images[0]
    # pipe.unet.load_attn_procs(model_id_or_path)

    uniquefilepath = uniquify("outputs/sd.jpg")
    image.save(uniquefilepath)

    if torch.cuda.is_available():
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        print("Upscaling by factor " + "4" + " using RealESRGAN")
        uniquefilepath = imageUpscaleRealESRGAN(uniquefilepath)
    return uploadToHoster((uniquefilepath))


def imageUpscaleRealESRGANUrl(url, upscale="4"):
    import requests
    from PIL import Image
    from io import BytesIO

    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image.save("temp.jpg")
    uniquefilepath = imageUpscaleRealESRGAN("temp.jpg", upscale)
    return uploadToHoster(uniquefilepath)


def imageUpscaleRealESRGAN(filepath, upscale="4"):
    import subprocess
    uniquefilepath = uniquify("outputs/sd.jpg")
    if upscale == "4":
        model = "realesrgan-x4plus"
    else:
        model = "realesr-animevideov3"
    FNULL = open(os.devnull, 'w')  # use this if you want to suppress output to stdout from the subprocess
    args = "tools\\realesrgan_upscaler\\realesrgan-ncnn-vulkan.exe -n " + model + " -s " + upscale + " -i " + filepath + " -o " + uniquefilepath
    subprocess.call(args, stdout=FNULL, stderr=FNULL, shell=False)
    # if os.path.isfile(filepath):
    #    os.remove(filepath)
    return uniquefilepath


def imageUpscale2x(url):
    from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline
    import torch
    import requests
    from PIL import Image
    from io import BytesIO

    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
    )
    pipeline.to("cuda")

    model_id = "stabilityai/sd-x2-latent-upscaler"
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    upscaler.to("cuda")

    prompt = "UHD, 4k, hyper realistic, extremely detailed, professional, vibrant, not grainy, smooth, sharp"
    generator = torch.manual_seed(33)

    response = requests.get(url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")

    upscaled_image = upscaler(
        prompt=prompt,
        image=low_res_img,
        num_inference_steps=20,
        guidance_scale=0,
        generator=generator,
    ).images[0]

    uniquefilepath = uniquify("outputs/sd.jpg")
    upscaled_image.save(uniquefilepath)

    if torch.cuda.is_available():
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return uploadToHoster(uniquefilepath)


def imageUpscale4x(url):
    import requests
    from PIL import Image
    from io import BytesIO
    from diffusers import StableDiffusionUpscalePipeline
    import torch

    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    # model_id = "stabilityai/sd-x2-latent-upscaler"
    pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()

    response = requests.get(url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = low_res_img.resize(
        (int(low_res_img.width / 2), int(low_res_img.height / 2)))  # This is bad but memory is too low.

    prompt = "UHD, 4k, hyper realistic, extremely detailed, professional, vibrant, not grainy, smooth, sharp"
    upscaled_image = pipe(prompt=prompt, image=low_res_img).images[0]
    uniquefilepath = uniquify("outputs/sd.jpg")
    upscaled_image.save(uniquefilepath)

    if torch.cuda.is_available():
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return uploadToHoster(uniquefilepath)


def GoogleTranslate(text, translation_lang):
    from translatepy.translators.google import GoogleTranslate
    gtranslate = GoogleTranslate()
    length = len(text)

    step = 0
    translated_text = ""
    if length > 4999:
        while step+5000 < length:
            textpart = text[step:step+5000]
            step = step + 5000
            try:
                translated_text_part = str(gtranslate.translate(textpart, translation_lang))
                print("Translated Text part:\n\n " + translated_text_part)
            except:
                translated_text_part = "An error occured"

            translated_text = translated_text + translated_text_part
        #go back to where we really are
        #step = step - 5000


    if step < length:
        textpart = text[step:length]
        try:
            translated_text_part = str(gtranslate.translate(textpart, translation_lang))
            print("Translated Text part:\n\n " + translated_text_part)
        except:
            translated_text_part = "An error occured"

        translated_text = translated_text + translated_text_part


    return translated_text


def OCRtesseract(url):
    import cv2
    import pytesseract
    import requests
    from PIL import Image
    from io import BytesIO

    if str(url).endswith("pdf"):
        result = extract_text_from_pdf(url)

    else:
        response = requests.get(url)
        imgd = Image.open(BytesIO(response.content)).convert("RGB")
        imgd.save("ocr.jpg")
        img = cv2.imread("ocr.jpg")
        img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((1, 1), np.uint8)
        # img = cv2.dilate(img, kernel, iterations=1)
        # img = cv2.erode(img, kernel, iterations=1)
        # img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cv2.imwrite("ocr_procesed.jpg", img)
        # img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Adding custom options
        custom_config = r'--oem 3 --psm 6'
        result = pytesseract.image_to_string(img, config=custom_config)
    print(str(result))
    return str(result)

def extract_text_from_pdf(url):
    from pypdf import PdfReader
    from pathlib import Path
    import requests
    file_path = Path('temp.pdf')
    response = requests.get(url)
    file_path.write_bytes(response.content)

    reader = PdfReader(file_path)
    number_of_pages = len(reader.pages)
    text = ""
    for page_num in range(number_of_pages):
        page = reader.pages[page_num]
        text = text + page.extract_text()


    return text


dict_users = {}
def LLAMA2(message, user, system_prompt="", clear=False):
    import requests
    import json

    if dict_users.get(user) is None:
        dict_users[user] = {'history': []}

    #get_promt_len(dict_users[user]['history'], message, system_prompt, "", "")

    result = ""
    split = message.split()
    length = min(len(split), 1800)
    for token in range(0, length):
        result = result + " " + split[token]

    message = result

    #print(str(dict_users[user]['history']))

    url = 'http://137.250.171.154:1337/assist'
    SYSTEM_PROMPT = "Your name is NostrDVM. You are a data vending machine, helping me support users with performing different AI tasks. If you don't know an answer, tell user to use the -help command for more info. Be funny."
    if system_prompt == "":
        system_prompt = SYSTEM_PROMPT
    DATA_DESC = ""
    DATA = ""

    payload = {
        "system_prompt": system_prompt,
        "data_desc": DATA_DESC,
        "data": DATA
    }

    def post_stream(url, data):
        s = requests.Session()  #
        answer = ""
        with s.post(url, json=json.dumps(data), stream=True) as resp:
            for line in resp:
                if line:
                    answer += line.decode()
        print(answer.lstrip())
        return answer.lstrip()

    if message == 'forget':
        dict_users[user]['history'] = []
        answer = "I have now forgotten about our chat history. Nice to meet you (again)."
    else:
        if clear:
            dict_users[user]['history'] = []

        payload['message'] = "User: " + message
        payload['history'] = dict_users[user]['history']
        answer = post_stream(url, payload)
        dict_users[user]['history'].append((message, answer))

    return answer




def NoteRecommendations(user, notactivesincedays, is_bot):
    from nostr_sdk import Keys, Client, Filter

    inactivefollowerslist = ""
    relay_list = ["wss://relay.damus.io", "wss://blastr.f7z.xyz", "wss://nostr-pub.wellorder.net", "wss://nos.lol", "wss://nostr.wine", "wss://relay.nostr.com.au", "wss://relay.snort.social"]
    relaytimeout = 5
    step = 20
    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    cl = Client(keys)
    for relay in relay_list:
        cl.add_relay(relay)
    cl.connect()

    timeinseconds = 3 * 24 * 60 * 60
    dif = Timestamp.now().as_secs() - timeinseconds
    considernotessince = Timestamp.from_secs(dif)
    filt = Filter().author(user).kind(1).since(considernotessince)
    reactions = cl.get_events_of([filt], timedelta(seconds=relaytimeout))
    list = []
    random.shuffle(reactions)
    for reaction in reactions:
        if reaction.kind() == 1:
            list.append(reaction.content())
    all = json.dumps(list)
    all = all.replace("\n", " ").replace("\n\n", " ")
    cleared = ""
    tokens = all.split()
    for item in tokens:
        item = item.replace("\n", " ").lstrip("\"").rstrip(",").rstrip(("."))
        #print(item)
        if item.__contains__("http") or item.__contains__("\nhttp")  or item.__contains__("\n\nhttp") or item.lower().__contains__("nostr:") or item.lower().__contains__("nevent") or item.__contains__("\\u"):
            cleareditem = ""
        else:
            cleareditem = item
        cleared = cleared + " " + cleareditem


    cleared = cleared.replace("\n", " ")
    #res = re.sub(r"[^ a-zA-Z0-9.!?/\\:,]+", '', all)
    #print(cleared)
    answer = LLAMA2("Give me the 15 most important substantives as keywords of the following input: " + cleared, "nostruser",
                       "Reply only with a comma-seperated keywords. return topics starting with a *", clear=True)


    promptarr = answer.split(":")
    if len(promptarr) > 1:
        #print(promptarr[1])
        prompt = promptarr[1].lstrip("\n").replace("\n", ",").replace("*", ",").replace("•", ",")
    else:
        prompt = promptarr[0].replace("\n", ",").replace("*", "")

    pattern = r"[^a-zA-Z,'\s]"
    text = re.sub(pattern, "", prompt) + ","

    #text = (text.replace("Let's,", "").replace("Why,", "").replace("GM,", "")
    #        .replace("Remember,", "").replace("I,", "").replace("Think,", "")
    #        .replace("Already,", ""))
    #print(text)
    keywords = text.split(', ')

    print(keywords)

    #answer = LLAMA2("Extent the given list with 5 synonyms per entry  " + str(keywords), user,
    #                "Reply only with a comma-seperated keywords. return topics starting with a *")
    #answer.replace(" - Alternatives:", ",")
    #print(answer)
    #promptarr = answer.split(":")
    #if len(promptarr) > 1:
    #    # print(promptarr[1])
    #    prompt = promptarr[1].lstrip("\n").replace("\n", ",").replace("*", "").replace("•", "")
    #else:
    #    prompt = promptarr[0].replace("\n", ",").replace("*", "")

    #pattern = r"[^a-zA-Z,'\s]"
    #text = re.sub(pattern, "", prompt) + ","
    #keywords = text.split(', ')

    #print(keywords)


    timeinseconds = 30 * 60  #last 30 min?
    dif = Timestamp.now().as_secs() - timeinseconds
    looksince = Timestamp.from_secs(dif)
    filt2 = Filter().kind(1).since(looksince)
    notes = cl.get_events_of([filt2], timedelta(seconds=10))

    #finallist = []
    finallist = {}

    print("Notes found: " + str(len(notes)))
    j=0
    for note in notes:
        j= j+1
        res = [ele for ele in keywords if(ele.replace(',',"") in note.content())]
        if bool(res):
            if not note.id().to_hex() in finallist and note.pubkey().to_hex() != user:
                finallist[note.id().to_hex()] = 0
                filt = Filter().kinds([9735, 7, 1]).event(note.id())
                reactions = cl.get_events_of([filt], timedelta(seconds=1))
                print(str(len(reactions)) + "   " + str(j) + "/" + str(len(notes)))
                finallist[note.id().to_hex()] = len(reactions)


    finallist_sorted = sorted(finallist.items(), key=lambda x: x[1], reverse=True)
    converted_dict = dict(finallist_sorted)
    print(json.dumps(converted_dict))

    notelist = ""
    resultlist = []
    i =0
    for k in converted_dict:
        #print(k)
        if is_bot:
            i = i+1
            notelist = notelist + "nostr:" + EventId.from_hex(k).to_bech32() + "\n\n"
            if i == 20:
                break
        else:
            p_tag = Tag.parse(["p", k])
            resultlist.append(p_tag.as_vec())

    if is_bot:
        return notelist
    else:
        return json.dumps(resultlist[:20])

# take second element for sort
def takeSecond(elem):
    return elem[1]

def InactiveNostrFollowers(user, notactivesincedays, is_bot):
    from nostr_sdk import Keys, Client, Filter

    inactivefollowerslist = ""
    relay_list = ["wss://relay.damus.io", "wss://blastr.f7z.xyz", "wss://nostr-pub.wellorder.net", "wss://nos.lol", "wss://nostr.wine", "wss://relay.nostr.com.au", "wss://relay.snort.social"]
    relaytimeout = 5
    step = 20
    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    cl = Client(keys)
    for relay in relay_list:
        cl.add_relay(relay)
    cl.connect()

    filt = Filter().author(user).kind(3).limit(1)
    followers = cl.get_events_of([filt], timedelta(seconds=relaytimeout))

    if len(followers) > 0:


        resultlist = []
        newest = 0
        bestentry = followers[0]
        for entry in followers:
            if entry.created_at().as_secs() > newest:
                newest = entry.created_at().as_secs()
                bestentry = entry

        i = 0
        print(bestentry.as_json())
        followings = []
        dic = {}
        for tag in bestentry.tags():
            if tag.as_vec()[0] == "p":
                following = tag.as_vec()[1]
                followings.append(following)
                dic[following] = "False"
        print("Followings: " + str(len(followings)))

        notactivesinceseconds = int(notactivesincedays) * 24 * 60 * 60
        dif = Timestamp.now().as_secs() - notactivesinceseconds
        notactivesince = Timestamp.from_secs(dif)

        while i < len(followings) - step:
            filters = []
            for i in range(i, i+step+1):
                filter1 = Filter().author(followings[i]).since(notactivesince).limit(1)
                filters.append(filter1)

            notes = cl.get_events_of(filters, timedelta(seconds=6))

            for note in notes:
                    dic[note.pubkey().to_hex()] = "True"
            print(str(i) + "/" + str(len(followings)))

        missing_scans = len(followings) - i
        filters = []
        for i in range(i+missing_scans):
            filter1 = Filter().author(followings[i]).since(notactivesince).limit(1)
            filters.append(filter1)

        notes = cl.get_events_of(filters, timedelta(seconds=6))
        for note in notes:
            dic[note.pubkey().to_hex()] = "True"

        print(str(len(followings)) + "/" + str(len(followings)))

        cl.disconnect()
        result = {k for (k, v) in dic.items() if v == "False"}
        if len(result) == 0:
            print("Not found")
            return "No inactive followers found on relays."

        for k in result:
            if (is_bot):

                inactivefollowerslist = inactivefollowerslist + "nostr:" + PublicKey.from_hex(k).to_bech32() + "\n"
            else:
                p_tag = Tag.parse(["p", k])
                resultlist.append(p_tag.as_vec())

        if (is_bot):
            return inactivefollowerslist
        else:
            return json.dumps(resultlist)

