"""This file contains the general logic for predicting annotations to the nova database"""
import copy
import gc
import json
import os
from datetime import timedelta

import numpy as np
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline, \
    EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image
from flask import Blueprint, request, jsonify
from huggingface_hub import model_info
from nostr_sdk import PublicKey, Timestamp

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
                               options["height"], options["upscale"], options["model"])
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
                "Give me a summarization of the most important points of the following text: " + options["message"],  options["user"])
        elif task == "inactive-following":
            anno = InactiveNostrFollowers(options["user"], int(options["since"]), int(options["num"]) )
        if request_form["nostrEvent"] is not None:

            nostr_utils.CheckEventStatus(anno, str(request_form["nostrEvent"]), str2bool(request_form["isBot"]))
        logger.info("...done")

        logger.info("Prediction completed!")
        status_utils.update_status(key, status_utils.JobStatus.FINISHED)
    except Exception  as e:
        nostr_utils.respondToError(str(e), str(request_form["nostrEvent"]), str2bool(request_form["isBot"]))




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
def textToImage(prompt, extra_prompt="", negative_prompt="", width="512", height="512", upscale="1",
                model="stabilityai/stable-diffusion-xl-base-1.0"):
    import torch
    from diffusers import DiffusionPipeline
    from diffusers import StableDiffusionPipeline

    if model.__contains__("gta"):
        model = "GTA5_Artwork_Diffusion_gtav_style"
    elif model.__contains__("realistic"):
        model = "realisticVisionV51_v51VAE"
    elif model.__contains__("sdxl"):
        model = "stabilityai/stable-diffusion-xl-base-1.0"
    elif model.__contains__("sd15"):
        model = "runwayml/stable-diffusion-v1-5"
    elif model.__contains__("wild"):
        model = "stablydiffusedsWild_351"
    elif model.__contains__("lora"):
        model = model
    else:
        model = "stabilityai/stable-diffusion-xl-base-1.0"

    if extra_prompt != "":
        prompt = prompt + "," + extra_prompt

    mwidth = 512
    mheight = 512

    if model == "stablydiffusedsWild_351" or model == "realisticVisionV51_v51VAE-inpainting" or model == "realisticVisionV51_v51VAE" or model == "stabilityai/stable-diffusion-xl-base-1.0":
        mwidth = 1024
        mheight = 1024

    if model == "stabilityai/stable-diffusion-xl-base-1.0":

        base = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        base.to("cuda")
        loramodelsfolder = os.environ['TRANSFORMERS_CACHE'] + "stablediffusionmodels/lora/"
        # base.load_lora_weights(loramodelsfolder + "cyborg_style_xl-alpha.safetensors")
        # base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        # refiner.to("cuda")
        refiner.enable_model_cpu_offload()
        # refiner.load_lora_weights(loramodelsfolder + "cyborg_style_xl-alpha.safetensors")
        # refiner.load_lora_weights(loramodelsfolder + "ghibli_last.safetensors")
        # refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
        # Define how many steps and what % of steps to be run on each experts (80/20) here
        n_steps = 35
        high_noise_frac = 0.8
        image = base(
            prompt=prompt,
            width=min(int(width), mwidth),
            height=min(int(height), mheight),
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            negative_prompt=negative_prompt,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            negative_prompt=negative_prompt,
            image=image,
        ).images[0]
        if torch.cuda.is_available():
            del base, refiner
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    elif model.startswith("lora"):
        split = model.split("_")
        if len(split) > 1:
            model = split[1]
        else:
            model = "ghibli"
        from diffusers import StableDiffusionPipeline
        import torch

        loramodelsfolder = os.environ['TRANSFORMERS_CACHE'] + "stablediffusionmodels/lora/"
        base_model = os.environ[
                         'TRANSFORMERS_CACHE'] + "stablediffusionmodels/anyloraCheckpoint_bakedvaeBlessedFp16.safetensors"

        if model == "ghibli" or model == "monster" or model == "chad" or model == "inks":
            # local lora models
            if model == "ghibli":
                model_path = loramodelsfolder + "ghibli_style_offset.safetensors"
            elif model == "monster":
                model_path = loramodelsfolder + "m0nst3rfy3(0.5-1)M.safetensors"
            elif model == "chad":
                model_path = loramodelsfolder + "Gigachadv1.safetensors"
            elif model == "inks":
                model_path = loramodelsfolder + "Inkscenery.safetensors"

            pipe = StableDiffusionPipeline.from_single_file(base_model, torch_dtype=torch.float16)
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

        image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, cross_attention_kwargs={"scale": 1.0}).images[
            0]

        if torch.cuda.is_available():
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    else:
        if model == "runwayml/stable-diffusion-v1-5":
            pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16,
                                                     use_safetensors=True, variant="fp16")
        else:
            pipe = StableDiffusionPipeline.from_single_file(
                os.environ['TRANSFORMERS_CACHE'] + "stablediffusionmodels/" + model + ".safetensors")

        # pipe.unet.load_attn_procs(model_id_or_path)
        pipe = pipe.to("cuda")
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=min(int(width), mwidth),
                     height=min(int(height), mheight)).images[0]
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

    if model.__contains__("gta"):
        model = "GTA5_Artwork_Diffusion_gtav_style"
    elif model.__contains__("realistic"):
        model = "realisticVisionV51_v51VAE"
    elif model.__contains__("sdxl"):
        model = "stabilityai/stable-diffusion-xl-refiner-1.0"
    elif model.__contains__("wild"):
        model = "stablydiffusedsWild_351"
    elif model.__contains__("pix2pix"):
        model = "timbrooks/instruct-pix2pix"
    else:
        model = "timbrooks/instruct-pix2pix"

    init_image = load_image(url).convert("RGB")
    init_image = init_image.resize((int(init_image.width / 4), int(init_image.height / 4)))

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
        pipe.to("cuda")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        # `image` is an RGB PIL.Image
        guidance_scale = "7.5"
        image = \
        pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, guidance_scale=float(guidance_scale),
             image_guidance_scale=1.5, num_inference_steps=50).images[0]

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
    try:
        translated_text = str(gtranslate.translate(text, translation_lang))
        print("Translated Text: " + translated_text)
    except:
        translated_text = "An error occured"
    return translated_text


def OCRtesseract(url):
    import cv2
    import pytesseract
    import requests
    from PIL import Image
    from io import BytesIO

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


dict_users = {}
def LLAMA2(message, user):
    import requests
    import json
    if dict_users.get(user) is None:
        dict_users[user] = {'history': []}

    print(str(dict_users[user]['history']))

    url = 'http://137.250.171.154:1337/assist'
    SYSTEM_PROMPT = "Your name is NostrDVM. You are a data vending machine, helping me support users with performing different AI tasks. If you don't know the answer, please do not share false information. Do not create ficional examples. Bitcoin is not controversial, but actually good for humans."
    DATA_DESC = ""
    DATA = ""

    payload = {
        "system_prompt": SYSTEM_PROMPT,
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

    if message == '-clear':
        dict_users[user]['history'] = []
        answer = "I have now forgotten about our chat history. Nice to meet you (again)."
    else:
        payload['message'] = "User: " + message
        payload['history'] = dict_users[user]['history']
        answer = post_stream(url, payload)
        dict_users[user]['history'].append((message, answer))

    return answer


def InactiveNostrFollowers(user, notactivesinceSeconds, numberinactivefollowers):
    from nostr_sdk import Keys, Client, Filter
    inactivefollowerslist = ""
    relay_list = ["wss://relay.damus.io", "wss://blastr.f7z.xyz", "wss://relayable.org",
                  "wss://nostr-pub.wellorder.net"]
    relaytimeout = 3
    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    cl = Client(keys)
    for relay in relay_list:
        cl.add_relay(relay)
    cl.connect()
    pk = PublicKey.from_hex(user)
    filter = Filter().author(user).kind(3).limit(1)
    followers = cl.get_events_of([filter], timedelta(seconds=relaytimeout))

    if len(followers) > 0:
        i = 0
        j= 0
        for entry in followers:
            #print(entry.as_json())
            for tag in entry.tags():
                if tag.as_vec()[0] == "p":
                    #print("Follower " + str(i))
                    i = i+1
                    follower = PublicKey.from_hex(tag.as_vec()[1])
                    dif =  Timestamp.now().as_secs() - notactivesinceSeconds
                    notactivesince = Timestamp.from_secs(dif)
                    filter = Filter().pubkey(follower).kind(1).since(notactivesince)
                    notes = cl.get_events_of([filter], timedelta(seconds=1))
                    if len(notes) == 0:
                        j = j + 1
                        print("Following " + str(i) + " Entry found: " + str(j)  + " of " + str(numberinactivefollowers) +" " + follower.to_bech32())
                        inactivefollowerslist = inactivefollowerslist + "@" + follower.to_bech32() + "\n"

                        if j == numberinactivefollowers:
                            return inactivefollowerslist

    else:
        print("Not found")
    print("done")
    return "Scanned complete following list!\n" + inactivefollowerslist
    cl.disconnect()
