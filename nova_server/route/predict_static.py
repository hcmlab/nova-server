"""This file contains the general logic for predicting annotations to the nova database"""
import copy
import gc
import json
import os

import cv2
import numpy as np
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline, \
    EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image
from flask import Blueprint, request, jsonify
from huggingface_hub import model_info

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
    options = []
    if request_form.get("optStr"):
        for k, v in [option.split("=") for option in request_form["optStr"].split(";")]:
            t = (k, v)
            options.append(v)
            logger.info(k + "=" + v)
    logger.info("...done.")

    #TODO move these to separate files
    if task == "text-to-image":
        anno = textToImage(options[0], options[1], options[2], options[3], options[4], options[5], options[6])
    elif task == "image-to-image":
        anno = imageToImage(options[0], options[1], options[2], options[3], options[4], options[5])
    elif task == "image-upscale":
        anno = imageUpscaleRealESRGANUrl(options[0], options[1])
    elif task == "translation":
        anno = GoogleTranslate(options[0], options[1])
    elif task == "ocr":
        anno = OCRtesseract(options[0])
    elif task == "chat":
        anno = FreeWilly(options[0])

    if request_form["nostrEvent"] is not None:
        nostr_utils.CheckEventStatus(anno, str(request_form["nostrEvent"]), str2bool(request_form["isBot"]))
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
def textToImage(prompt, extra_prompt="",  negative_prompt="", width="512", height="512", upscale="1", model="stablydiffusedsWild_351"):
    import torch
    from diffusers import DiffusionPipeline
    from diffusers import StableDiffusionPipeline

    if extra_prompt != "":
        prompt = prompt + "," + extra_prompt

    mwidth = 512
    mheight = 512

    if model == "stablydiffusedsWild_351" or model == "realisticVisionV51_v51VAE-inpainting" or model == "realisticVisionV51_v51VAE" or model == "stabilityai/stable-diffusion-xl-base-1.0":
        mwidth = 1024
        mheight = 1024

    elif model == "GTA5_Artwork_Diffusion_gtav_style" or "runwayml/stable-diffusion-v1-5":
        mwidth = 512
        mheight = 512




    if model == "stabilityai/stable-diffusion-xl-base-1.0":

        base = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        base.to("cuda")
        loramodelsfolder = os.environ['TRANSFORMERS_CACHE'] + "stablediffusionmodels/lora/"
        #base.load_lora_weights(loramodelsfolder + "cyborg_style_xl-alpha.safetensors")
        #base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        #refiner.to("cuda")
        refiner.enable_model_cpu_offload()
        #refiner.load_lora_weights(loramodelsfolder + "cyborg_style_xl-alpha.safetensors")
        #refiner.load_lora_weights(loramodelsfolder + "ghibli_last.safetensors")
        #refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
        # Define how many steps and what % of steps to be run on each experts (80/20) here
        n_steps = 35
        high_noise_frac = 0.8
        image = base(
            prompt=prompt,
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
        base_model = os.environ[ 'TRANSFORMERS_CACHE'] + "stablediffusionmodels/anyloraCheckpoint_bakedvaeBlessedFp16.safetensors"

        if model == "ghibli" or model == "monster" or model == "chad" or model == "inks":
            #local lora models
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
            #huggingface repo lora models
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

        image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, cross_attention_kwargs={"scale": 1.0}).images[0]

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
            pipe = StableDiffusionPipeline.from_single_file(os.environ['TRANSFORMERS_CACHE'] + "stablediffusionmodels/" + model + ".safetensors")

        # pipe.unet.load_attn_procs(model_id_or_path)
        pipe = pipe.to("cuda")
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=min(int(width), mwidth), height=min(int(height), mheight) ).images[0]
        if torch.cuda.is_available():
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


    uniquefilepath = uniquify("outputs/sd.jpg")
    image.save(uniquefilepath)


    if(int(upscale) > 1 and int(upscale) <= 4):
        print("Upscaling by factor " + upscale + " using RealESRGAN")
        uniquefilepath = imageUpscaleRealESRGAN(uniquefilepath, upscale)

    return uploadToHoster(uniquefilepath)


def imageToImage(url, prompt, negative_prompt, strength, guidance_scale, model="stablydiffusedsWild_351"):
    import requests
    import torch
    from PIL import Image
    from io import BytesIO

    from diffusers import StableDiffusionImg2ImgPipeline



    response = requests.get(url)
    init_image = load_image(url).convert("RGB")
    #init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((int(init_image.width/4), int(init_image.height/4)))

    device = "cuda"
    if model == "stabilityai/stable-diffusion-xl-refiner-1.0":
        #init_image = init_image.resize((int(init_image.width / 2), int(init_image.height / 2)))
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model, torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True
        )
        pipe = pipe.to("cuda")
        image = pipe(prompt, image=init_image, strength=float(strength),guidance_scale=float(guidance_scale)).images[0]
    elif model == "timbrooks/instruct-pix2pix":
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model, torch_dtype=torch.float16,
                                                                      safety_checker=None)
        pipe.to("cuda")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        # `image` is an RGB PIL.Image
        guidance_scale="7.5"
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image, guidance_scale=float(guidance_scale),  image_guidance_scale=1.5, num_inference_steps=50).images[0]

        # 'CompVis/stable-diffusion-v1-4'
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            os.environ['TRANSFORMERS_CACHE'] + "stablediffusionmodels/"+model+".safetensors"
        )
        pipe = pipe.to(device)

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
    args = "tools\\realesrgan_upscaler\\realesrgan-ncnn-vulkan.exe -n " + model +" -s " + upscale + " -i "+ filepath +  " -o " + uniquefilepath
    subprocess.call(args, stdout=FNULL, stderr=FNULL, shell=False)
    #if os.path.isfile(filepath):
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
    #model_id = "stabilityai/sd-x2-latent-upscaler"
    pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()


    response = requests.get(url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = low_res_img.resize((int(low_res_img.width/2), int(low_res_img.height/2)))  # This is bad but memory is too low.

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
    #img = cv2.dilate(img, kernel, iterations=1)
    #img = cv2.erode(img, kernel, iterations=1)
    #img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite("ocr_procesed.jpg", img)
    #img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]



    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    result = pytesseract.image_to_string(img, config=custom_config)
    print(str(result))
    return str(result)

def FreeWilly(message):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/FreeWilly2", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained("stabilityai/FreeWilly2", torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True, device_map="cuda")

    system_prompt = "### System:\nYou are Free Willy, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"


    prompt = f"{system_prompt}### User: {message}\n\n### Assistant:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)

    print(tokenizer.decode(output[0], skip_special_tokens=True))

    return tokenizer.decode(output[0], skip_special_tokens=True)
