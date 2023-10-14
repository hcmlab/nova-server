"""This file contains the general logic for predicting annotations to the nova database"""
import gc
import json
import os
import re
from datetime import timedelta
import random
from threading import Thread

import numpy as np
from compel import ReturnedEmbeddingsType
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline, \
    EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, StableDiffusionXLInpaintPipeline, \
    StableDiffusionXLPipeline
from diffusers.utils import load_image
from flask import Blueprint, request, jsonify
from huggingface_hub import model_info
from nostr_sdk import PublicKey, Timestamp, Tag, EventId
from safetensors import torch

from nova_server.utils.thread_utils import THREADS
from nova_server.utils.status_utils import update_progress
from nova_server.utils.key_utils import get_key_from_request_form, str2bool
from nova_server.utils import (
    thread_utils,
    status_utils,
    log_utils,
    nostr_dvm
)
from hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable import (
    HcaiNovaDynamicIterable,
)
from nova_utils.interfaces.server_module import Trainer as iTrainer

os.environ['TRANSFORMERS_CACHE'] = 'W:/nova/cml/models/trainer/.cache/'

predict_static = Blueprint("predict_static", __name__)
wait_for_send = True

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
            anno = textToImage(options["prompt"], options["extra_prompt"], options["negative_prompt"],
                               options["upscale"], options["model"],  options["ratiow"], options["ratioh"], options["lora"] , options["lora_weight"])
        elif task == "image-to-image":
            anno = imageToImage(options["url"], options["prompt"], options["negative_prompt"], options["strength"],
                                options["guidance_scale"], options["model"], options["lora"] , options["lora_weight"])
        elif task == "image-reimagine":
            anno = imageReimagine(options["url"])
        elif task == "image-upscale":
            anno = imageUpscaleRealESRGANUrl(options["url"], options["upscale"])
        elif task == "translation":
            anno = GoogleTranslate(options["text"], options["translation_lang"])

        elif task == "image-to-text":
            if options["kind"] == "prompt":
                anno = ImageToPrompt(options["url"])
            else:
                 anno = OCRtesseract(options["url"])
        elif task == "chat":
            anno = LLAMA2(options["message"], options["user"])
        elif task == "summarization":
            anno = LLAMA2(
                "Give me a summarization of the most important points of the following text: " + options["message"],  options["user"], options["system_prompt"])
        elif task == "inactive-following":
            anno = InactiveNostrFollowers(options["user"], int(options["since"]), str2bool(options["is_bot"]), request_form["dvmkey"])
        elif task == "note-recommendation":
            anno = NoteRecommendations(options["user"], int(options["since"]), str2bool(options["is_bot"]))



        logger.info("...done")
        logger.info("Prediction completed!")
        status_utils.update_status(key, status_utils.JobStatus.FINISHED)
        if "nostrEvent" in request_form:
            if request_form["nostrEvent"] is not None:
                nostr_dvm.check_event_status(data=anno, original_event_str=str(request_form["nostrEvent"]), dvm_key=request_form["dvmkey"],
                                             use_bot=str2bool(request_form["isBot"]))

    except Exception  as e:
        if "nostrEvent" in request_form:
            nostr_dvm.respond_to_error(content=str(e), originaleventstr=str(request_form["nostrEvent"]), is_from_bot=str2bool(request_form["isBot"]),
                                                                dvm_key=request_form["dvmkey"])




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
def textToImage(prompt, extra_prompt="", negative_prompt="", upscale="1",
                model="stabilityai/stable-diffusion-xl-base-1.0", ratiow="1", ratioh="1", lora="", lora_weight=""):
    import torch
    from diffusers import DiffusionPipeline
    from diffusers import StableDiffusionPipeline
    from compel import Compel



    if negative_prompt == "":
        negative_prompt = 'bad anatomy, bad hands, deformed face, weird eyes, lowres, error, missing fingers, worst quality, low quality, jpeg artifacts, signature, watermark'


    model = model.rstrip(" ")
    if model.__contains__("realistic"):
        model = "realisticVisionV51_v51VAE"
    elif model.__contains__("sd15"):
        model = "runwayml/stable-diffusion-v1-5"
    elif model.__contains__("sd21"):
        model = "stabilityai/stable-diffusion-2-1"
    elif model.__contains__("wild"):
        model = "stablydiffusedsWild_351"
    elif model.startswith("lora"):
        model = model #handle sd15 lora models seperatley
    elif model.__contains__("dreamshaper"):
        model = os.environ[
                    'TRANSFORMERS_CACHE'] + "stablediffusionmodels/dreamshaperXL.safetensors"
    elif model.__contains__("nightvision"):
        model = os.environ[
                    'TRANSFORMERS_CACHE'] + "stablediffusionmodels/nightvisionXL.safetensors"
    elif model.__contains__("protovision"):
        model = os.environ[
                    'TRANSFORMERS_CACHE'] + "stablediffusionmodels/protovisionXL.safetensors"
    elif model.__contains__("dynavision"):
        model = os.environ[
                    'TRANSFORMERS_CACHE'] + "stablediffusionmodels/dynavisionXL.safetensors"
    elif model.__contains__("sdvn"):
        model = os.environ[
                'TRANSFORMERS_CACHE'] + "stablediffusionmodels/sdvn6Realxl_detailface.safetensors"
    elif model.__contains__("fantastic"):
        model = os.environ[
                    'TRANSFORMERS_CACHE'] + "stablediffusionmodels/fantasticCharacters_v55.safetensors"
    elif model.__contains__("chroma"):
        model = os.environ[
                    'TRANSFORMERS_CACHE'] + "stablediffusionmodels/zavychromaxl_v10.safetensors"
    elif model.__contains__("crystalclear"):
        model = os.environ[
                    'TRANSFORMERS_CACHE'] + "stablediffusionmodels/crystalClearXL.safetensors"
    elif model.__contains__("juggernaut"):
        model = os.environ[
                    'TRANSFORMERS_CACHE'] + "stablediffusionmodels/juggernautXL_version2.safetensors"
    elif model.__contains__("timeless"):
        model = os.environ[
                    'TRANSFORMERS_CACHE'] + "stablediffusionmodels/copaxTimelessxlSDXL1_v5.safetensors"
    elif model.__contains__("realisticfreedom"):
        model = os.environ[
                    'TRANSFORMERS_CACHE'] + "stablediffusionmodels/realisticFreedomSFW_alpha.safetensors"
    elif model.__contains__("fodda"):
        model = os.environ[
                    'TRANSFORMERS_CACHE'] + "stablediffusionmodels/foddaxlPhotorealism_v35.safetensors"
    elif model.__contains__("realvis"):
        model = os.environ[
                    'TRANSFORMERS_CACHE'] + "stablediffusionmodels/realvisxlV10_v10VAE.safetensors"



    else:
        model = "stabilityai/stable-diffusion-xl-base-1.0"

    if extra_prompt != "":
        prompt = prompt + "," + extra_prompt

    print(model)
    mwidth = 768
    mheight = 768

    if model.startswith("lora"):
        mwidth = 768
        mheight = 768


    if model == "stabilityai/stable-diffusion-xl-base-1.0" or  model.__contains__("dreamshaperXL") or model.__contains__("nightvisionXL") or model.__contains__("protovisionXL") or model.__contains__("dynavisionXL") or model.__contains__("sdvn6Realxl_detailface") or model.__contains__("fantasticCharacters_v55")  or model.__contains__("zavychromaxl_v10") or model.__contains__("crystalClearXL") or model.__contains__("juggernautXL_version2") or model.__contains__("copaxTimelessxlSDXL1_v5") or model.__contains__("realisticFreedomSFW_alpha") or model.__contains__("foddaxlPhotorealism_v35") or model.__contains__("realvisxlV10_v10VAE"):
        mwidth = 1024
        mheight = 1024

    height = mheight
    width = mwidth

    ratiown = int(ratiow)
    ratiohn= int(ratioh)

    if ratiown > ratiohn:
        height = int((ratiohn/ratiown) * float(width))
    elif ratiown < ratiohn:
        width = int((ratiown/ratiohn) * float(height))
    elif ratiown == ratiohn:
        width = height

    print("Output width: " + str(width) + " Output height: " + str(height))

    if (model == "stabilityai/stable-diffusion-xl-base-1.0" or model.__contains__("dreamshaperXL") or model.__contains__(
            "nightvisionXL") or model.__contains__("protovisionXL") or model.__contains__(
            "dynavisionXL") or model.__contains__("sdvn6Realxl_detailface") or model.__contains__("fantasticCharacters_v55")
            or model.__contains__("zavychromaxl_v10") or model.__contains__("crystalClearXL") or model.__contains__("juggernautXL_version2") or model.__contains__("copaxTimelessxlSDXL1_v5")  or model.__contains__("realisticFreedomSFW_alpha")  or model.__contains__("foddaxlPhotorealism_v35") or model.__contains__("realvisxlV10_v10VAE")):
        print("Loading model...")
        if model == "stabilityai/stable-diffusion-xl-base-1.0":

            base = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
            print("Loaded model: " + model)

        else:
            base = StableDiffusionXLPipeline.from_single_file(model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
            print("Loaded model: " + model)

        existing_lora = False
        if lora != "":
            print("Loading lora...")
            lora_models_folder = os.environ['TRANSFORMERS_CACHE'] + "stablediffusionmodels/lora/"
            lora, prompt, existing_lora =  buildloraxl(lora, prompt, lora_weight)

            if existing_lora:
                lora_weights = lora_models_folder + lora + ".safetensors"
                base.load_lora_weights(lora_weights)
                print("Loaded Lora: " + lora_weights)



        # base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        compel_base = Compel(
            tokenizer=[base.tokenizer, base.tokenizer_2],
            text_encoder=[base.text_encoder, base.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

        compel_refiner = Compel(
            tokenizer=[refiner.tokenizer_2],
            text_encoder=[refiner.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[True],
        )

        conditioning, pooled = compel_base(prompt)
        negative_conditioning, negative_pooled = compel_base(negative_prompt)

        conditioning_refiner, pooled_refiner = compel_refiner(prompt)
        negative_conditioning_refiner, negative_pooled_refiner = compel_refiner(
            negative_prompt
        )

        #refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
        # Define how many steps and what % of steps to be run on each experts (80/20) here

        n_steps = 35
        high_noise_frac = 0.8
        img = base(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=negative_conditioning,
            negative_pooled_prompt_embeds=negative_pooled,
            width=width,
            height=height,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
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
            prompt_embeds=conditioning_refiner,
            pooled_prompt_embeds=pooled_refiner,
            negative_prompt_embeds=negative_conditioning_refiner,
            negative_pooled_prompt_embeds=negative_pooled_refiner,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            num_images_per_prompt = 1,
            image=img,
        ).images[0]

        if torch.cuda.is_available():
            del refiner
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

        lora_models_folder = os.environ['TRANSFORMERS_CACHE'] + "stablediffusionmodels/lora/"
        base_model = os.environ[
                         'TRANSFORMERS_CACHE'] + "stablediffusionmodels/anyloraCheckpoint_bakedvaeBlessedFp16.safetensors"

        model_path = lora_models_folder + "Inkscenery.safetensors"
        if model == "inks" or model == "pepe" or model == "journey" or model == "ghibli" or model == "gigachad" or model == "food" or model == "zeus":
            if model == "inks":
                # local lora models
                model_path = lora_models_folder + "Inkscenery.safetensors"
                base_model = os.environ[
                                 'TRANSFORMERS_CACHE'] + "stablediffusionmodels/dreamshaper_8.safetensors"
                prompt = "white background, scenery, ink, mountains, water, trees, " + str(prompt).lstrip() + " <lora:ink-0.1-3-b28-bf16-D128-A1-1-ep64-768-DAdaptation-cosine:1>"


            elif model == "journey":
                # local lora models
                model_path = lora_models_folder + "OpenJourney-LORA.safetensors"
                prompt = "<lora:openjourneyLora_v1:1> " + prompt

            elif model == "ghibli":
                # local lora models
                model_path = lora_models_folder + "ghibli_style_offset.safetensors"
                prompt = "<lora:ghibli_style_offset:1> " + prompt

            elif model == "gigachad":
                # local lora models
                model_path = lora_models_folder + "Gigachadv1.safetensors"
                prompt = "Gigachad, " + prompt + " <lora:Gigachadv1:0.8>"

            elif model == "food":
                # local lora models
                model_path = lora_models_folder + "foodphoto.safetensors"
                prompt = "RAW photo, " + prompt + ", foodphoto <lora:foodphoto:1>"

            elif model == "zeus":
                base_model = os.environ[
                                 'TRANSFORMERS_CACHE'] + "stablediffusionmodels/revAnimated_v122EOL.safetensors"
                model_path = lora_models_folder + "zeus.safetensors"
                prompt = "Z3US, " + prompt
                negative_prompt = ("glass, pedestal, socle, basement, camera, subsurface, underwater, hypermaximalist,"
                                   " diamond, (flowers), water, (leaves) (worst quality:2), (low quality:2), "
                                   "(normal quality:2), border, frame, poorly drawn, childish,((dof)),bad hands, "
                                   "ugly hands, unnatural hands ") + negative_prompt
            elif model == "pepe":
                model_path = lora_models_folder + "pepe_frog_v2.safetensors"
                base_model = os.environ[
                                 'TRANSFORMERS_CACHE'] + "stablediffusionmodels/deliberate_v2.safetensors"
                prompt = "pepe_frog, " + prompt + "  <lora:pepe_frog_v2:1>"
                negative_prompt = "rz-neg-15-foranalog verybadimagenegative_v1., " + negative_prompt

            pipe = StableDiffusionPipeline.from_single_file(base_model, torch_dtype=torch.float16, safety_checker = None)

            print("Loaded lora: " + model_path)
            print("Loaded model: " + base_model)
            pipe.safety_checker = None
            pipe.requires_safety_checker = False
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.to("cuda")

            pipe.load_lora_weights(model_path)
            #pipe.unet.load_attn_procs(model_path)

            print(prompt)
            image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=30, width=width,
                         height=height, guidance_scale=7.5, cross_attention_kwargs={"scale": 1.0}).images[0]

            if torch.cuda.is_available():
                del pipe
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    else:
        print("entering else case")
        if model == "runwayml/stable-diffusion-v1-5":
            pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16,
                                                     use_safetensors=True, variant="fp16"
                                                     )
        else:
            pipe = StableDiffusionPipeline.from_single_file(
                os.environ['TRANSFORMERS_CACHE'] + "stablediffusionmodels/" + model + ".safetensors",
                safety_checker = None, requires_safety_checker = False)

        # pipe.unet.load_attn_procs(model_id_or_path)
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
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

    result_url = uploadToHoster(uniquefilepath)

    return result_url
def imageReimagine(url):
    # pip install git+https://github.com/huggingface/diffusers.git transformers accelerate
    import requests
    import torch
    from PIL import Image
    from io import BytesIO

    from diffusers import StableUnCLIPImg2ImgPipeline

    # Start the StableUnCLIP Image variations pipeline
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # Get image from URL
    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")

    # Pipe to make the variation
    images = pipe(init_image).images
    uniquefilepath = uniquify("outputs/sd.jpg")
    images[0].save(uniquefilepath)

    if torch.cuda.is_available():
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        print("Upscaling by factor " + "4" + " using RealESRGAN")
        uniquefilepath = imageUpscaleRealESRGAN(uniquefilepath)
    return uploadToHoster(uniquefilepath)


def buildloraxl(lora, prompt, lora_weight):
    existing_lora = False
    if lora == "cyborg_style_xl":
        if lora_weight == "":
            lora_weight = "0.8"
        prompt = " cyborg style, " + prompt + " <lora:cyborg_style_xl:"+lora_weight+">" #0.8
        existing_lora = True

    if lora == "3d_render_style_xl":
        if lora_weight == "":
            lora_weight = "1"
        prompt = "3d style, 3d render, " + prompt + " <lora:3d_render_style_xl:"+lora_weight+">"
        existing_lora = True

    if lora == "psychedelic_noir_xl":
        if lora_weight == "":
            lora_weight = "1"
        prompt = prompt + " <lora:Psychedelic_Noir__sdxl:"+lora_weight+">>"
        existing_lora = True

    if lora == "wojak_xl":
        if lora_weight == "":
            lora_weight = "1"
        prompt = "<lora:wojak_big:"+lora_weight+">, " + prompt + ", wojak"
        existing_lora = True

    if lora == "dreamarts_xl":
        if lora_weight == "":
            lora_weight = "1"
        prompt = "<lora:DreamARTSDXL:"+lora_weight+">, " + prompt
        existing_lora = True

    if lora == "voxel_xl":
        if lora_weight == "":
            lora_weight = "1"
        prompt = "voxel style, " + prompt + " <lora:last:"+lora_weight+">"
        existing_lora = True

    if lora == "kru3ger_xl":
        if lora_weight == "":
            lora_weight = "1"
        prompt = "kru3ger_style, " + prompt + "<lora:sebastiankrueger-kru3ger_style-000007:"+lora_weight+">"
        existing_lora = True

    if lora == "ink_punk_xl":
        if lora_weight == "":
            lora_weight = "0.5"
        prompt = "inkpunk style, " + prompt + "  <lora:IPXL_v2:"+lora_weight+">"
        existing_lora = True

    if lora == "real_art_xl":
        if lora_weight == "":
            lora_weight = "0.8"
        prompt = prompt + " <lora:xl_real_beta1:"+lora_weight+">"
        existing_lora = True

    if lora == "more_art_xl":
        if lora_weight == "":
            lora_weight = "1"
        prompt = prompt + " <lora:xl_more_art-full_v1:"+lora_weight+">"
        existing_lora = True

    if lora == "2077_style_xl":
        if lora_weight == "":
            lora_weight = "1"
        prompt = "cyberpunk, " + prompt + " <lora:2077_Style:"+lora_weight+">"
        existing_lora = True

    if lora == "ink_scenery_xl":
        if lora_weight == "":
            lora_weight = "1"
        prompt = " ink scenery, " + prompt + " <lora:ink_scenery_xl:"+lora_weight+">"
        existing_lora = True

    if lora == "pixel_art_xl":
        if lora_weight == "":
            lora_weight = "1"
        prompt = "<lora:pixelbuildings128-v1:"+lora_weight+"> pixel art, " + prompt
        existing_lora = True

    if lora == "crayons_xl":
        if lora_weight == "":
            lora_weight = "1"
        prompt = "<lora:crayons_v1_sdxl:"+lora_weight+"> " + prompt
        existing_lora = True

    if lora == "watercolor_xl":
        if lora_weight == "":
            lora_weight = "1"
        prompt = "<lora:watercolor_v1_sdxl:"+lora_weight+"> " + prompt
        existing_lora = True

    return lora, prompt, existing_lora


def ImageToPrompt(url):
    from PIL import Image
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    kind =  "prompt" #"analysis" #prompt
    mode = "fast"
    init_image = load_image(url).convert("RGB")
    mwidth = 256
    mheight = 256


    w = mwidth
    h = mheight
    if init_image.width > init_image.height:
        scale = float(init_image.height / init_image.width)
        w = mwidth
        h = int(mheight * scale)
    elif init_image.width < init_image.height:
        scale = float(init_image.width / init_image.height)
        w = int(mwidth * scale)
        h = mheight
    else:
        w = mwidth
        h = mheight

    init_image = init_image.resize((w, h))

    from clip_interrogator import Config, Interrogator

    config = Config(clip_model_name="ViT-L-14/openai", device="cpu")


    if kind == "analysis":
        ci = Interrogator(config)


        image_features = ci.image_to_features(init_image)

        top_mediums = ci.mediums.rank(image_features, 5)
        top_artists = ci.artists.rank(image_features, 5)
        top_movements = ci.movements.rank(image_features, 5)
        top_trendings = ci.trendings.rank(image_features, 5)
        top_flavors = ci.flavors.rank(image_features, 5)

        medium_ranks = {medium: sim for medium, sim in zip(top_mediums, ci.similarities(image_features, top_mediums))}
        artist_ranks = {artist: sim for artist, sim in zip(top_artists, ci.similarities(image_features, top_artists))}
        movement_ranks = {movement: sim for movement, sim in
                          zip(top_movements, ci.similarities(image_features, top_movements))}
        trending_ranks = {trending: sim for trending, sim in
                          zip(top_trendings, ci.similarities(image_features, top_trendings))}
        flavor_ranks = {flavor: sim for flavor, sim in zip(top_flavors, ci.similarities(image_features, top_flavors))}

        result = "Medium Ranks:\n" + str(medium_ranks) + "\nArtist Ranks: " + str(artist_ranks) +  "\nMovement Ranks:\n" + str(movement_ranks) + "\nTrending Ranks:\n" + str(trending_ranks) +  "\nFlavor Ranks:\n" + str(flavor_ranks)

        print(result)
        return result
    else:

        ci = Interrogator(config)
        ci.config.blip_num_beams = 64
        ci.config.chunk_size = 2024
        ci.config.clip_offload = True
        ci.config.apply_low_vram_defaults()
        #MODELS = ['ViT-L (best for Stable Diffusion 1.*)']
        ci.config.flavor_intermediate_count = 2024 #if clip_model_name == MODELS[0] else 1024

        image = init_image
        if mode == 'best':
            prompt = ci.interrogate(image)
        elif mode == 'classic':
            prompt = ci.interrogate_classic(image)
        elif mode == 'fast':
            prompt = ci.interrogate_fast(image)
        elif mode == 'negative':
            prompt = ci.interrogate_negative(image)

        return prompt,

    #return medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks

   # config = Config(clip_model_name=os.environ['TRANSFORMERS_CACHE'] + "ViT-L-14/openai", device="cuda")git
   # ci = Interrogator(config)
    # "ViT-L-14/openai"))
    # "ViT-g-14/laion2B-s34B-b88K"))
    print("after function")

    #detected = ci.interrogate(init_image)
    print("after interogate")
    #print(str(detected))
    detected = "No worky"
    return str(detected)




def imageToImage(url, prompt, negative_prompt, strength, guidance_scale, model="pix2pix", lora="", lora_weight=""):
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline
    from diffusers.utils import load_image

    mwidth = 768
    mheight = 768


    if model.__contains__("realistic"):
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
        model = "stabilityai/stable-diffusion-xl-refiner-1.0"

    init_image = load_image(url).convert("RGB")


    if  model == "dreamshaper_8" or model == "stabilityai/stable-diffusion-xl-refiner-1.0":
        mwidth = 1024
        mheight = 1024



    w = mwidth
    h = mheight
    if init_image.width > init_image.height:
        scale = float(init_image.height / init_image.width)
        w = mwidth
        h = int(mheight * scale)
    elif init_image.width < init_image.height:
        scale = float(init_image.width / init_image.height)
        w = int(mwidth * scale)
        h = mheight
    else:
        w = mwidth
        h = mheight

    init_image = init_image.resize((w, h))


    if lora != "":
        print("Loading lora...")
        lora_models_folder = os.environ['TRANSFORMERS_CACHE'] + "stablediffusionmodels/lora/"
        lora, prompt, existing_lora = buildloraxl(lora, prompt, lora_weight)

        if existing_lora:
            model = "sdxl"


    if model == "sdxl":
        from diffusers import AutoPipelineForImage2Image
        import torch



        #init_image = init_image.resize((int(w/2), int(h/2)))

        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16).to("cuda")

        if existing_lora:
            lora_weights = lora_models_folder + lora + ".safetensors"
            pipe.load_lora_weights(lora_weights)
            print("Loaded Lora: " + lora_weights)

        seed = 20000
        generator = torch.manual_seed(seed)

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
       
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            generator=generator,
            strength=0.4).images[0]


    elif model == "stabilityai/stable-diffusion-xl-refiner-1.0":

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model, torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True
        )



        n_steps = 30
        high_noise_frac = 0.75
        transformation_strength = float(strength)
        cfg_scale = float(guidance_scale)
        transformation_strength = 0.58
        cfg_scale = 11.0
        #

        #prompt = prompt + ", " + ImageToPrompt(url)
        pipe = pipe.to("cuda")
        image = pipe(prompt, image=init_image,
                     negative_prompt=negative_prompt, num_inference_steps=n_steps, strength=transformation_strength, guidance_scale=cfg_scale).images[0]


    elif model == "timbrooks/instruct-pix2pix":
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model, torch_dtype=torch.float16,
                                                                      safety_checker=None)

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        pipe.to("cuda")
        image = pipe(prompt, negative_prompt=negative_prompt, image=init_image, num_inference_steps=10, image_guidance_scale=1).images[0]
        #image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_image).images[0]

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
        #model = "realesrgan-x4plus"
        model= "4x-UltraSharp-opt-fp16"
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
    from nostr_sdk import Keys, Client, Filter, Options
    from types import SimpleNamespace

    ns = SimpleNamespace()
    inactivefollowerslist = ""
    relay_list = ["wss://relay.damus.io", "wss://nostr-pub.wellorder.net", "wss://nos.lol", "wss://nostr.wine",
                  "wss://relay.nostfiles.dev", "wss://nostr.mom", "wss://nostr.oxtr.dev", "wss://relay.nostr.bg", "wss://relay.f7z.io"]
    relaytimeout = 5
    step = 20
    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    opts = Options().wait_for_send(wait_for_send).send_timeout(timedelta(seconds=5)).skip_disconnected_relays(True)
    cl = Client.with_opts(keys, opts)
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

    pattern = r"[^a-zA-Z,#'\s]"
    text = re.sub(pattern, "", prompt) + ","

    #text = (text.replace("Let's,", "").replace("Why,", "").replace("GM,", "")
    #        .replace("Remember,", "").replace("I,", "").replace("Think,", "")
    #        .replace("Already,", ""))
    #print(text)
    keywords = text.split(', ')
    keywords = [x.lstrip().rstrip(',') for x in keywords if x]


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
    notes = cl.get_events_of([filt2], timedelta(seconds=6))

    #finallist = []
    ns.finallist = {}

    print("Notes found: " + str(len(notes)))

    def scanList(noteid: EventId, instance, i, length):

        relay_list = ["wss://relay.damus.io", "wss://nostr-pub.wellorder.net", "wss://nos.lol", "wss://nostr.wine",
                      "wss://relay.nostfiles.dev", "wss://nostr.mom", "wss://nostr.oxtr.dev", "wss://relay.nostr.bg", "wss://relay.f7z.io"]
        keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
        opts = Options().wait_for_send(wait_for_send).send_timeout(timedelta(seconds=5)).skip_disconnected_relays(True)
        cli = Client.with_opts(keys, opts)
        for relay in relay_list:
            cli.add_relay(relay)
        cli.connect()

        filters = []
        instance.finallist[noteid.to_hex()] = 0
        filt = Filter().kinds([9735, 7, 1]).event(noteid)
        reactions = cl.get_events_of([filt], timedelta(seconds=5))
        print(str(len(reactions)) + "   " + str(j) + "/" + str(len(notes)))
        instance.finallist[noteid.to_hex()] = len(reactions)

        print(str(i) + "/" + str(length))
        cli.disconnect()


    j=0

    threads = []

    for note in notes:


        j= j+1
        res = [ele for ele in keywords if(ele.replace(',',"") in note.content())]
        if bool(res):
            if not note.id().to_hex() in ns.finallist and note.pubkey().to_hex() != user:
                args = [note.id(), ns, j, len(notes)]
                t = Thread(target=scanList, args=args)
                threads.append(t)

        # Start all threads
    for x in threads:
        x.start()

        # Wait for all of them to finish
    for x in threads:
        x.join()

    finallist_sorted = sorted(ns.finallist.items(), key=lambda x: x[1], reverse=True)
    converted_dict = dict(finallist_sorted)
    print(json.dumps(converted_dict))

    notelist = ""
    resultlist = []
    i =0
    notelist = "Based on topics: " + json.dumps(keywords).lstrip("[").rstrip(("]"))+ "\n\n"
    for k in converted_dict:
        #print(k)
        if is_bot:
            i = i+1
            notelist = notelist + "nostr:" + EventId.from_hex(k).to_bech32() + "\n\n"
            if i == 25:

                break
        else:
            p_tag = Tag.parse(["p", k])
            resultlist.append(p_tag.as_vec())

    if is_bot:
        return notelist
    else:
        return json.dumps(resultlist[:25])

# take second element for sort
def takeSecond(elem):
    return elem[1]

def InactiveNostrFollowers(user, notactivesincedays, is_bot, dvmkey):
    from nostr_sdk import Keys, Client, Filter, Options
    from types import SimpleNamespace

    ns = SimpleNamespace()

    detailscan = False
    inactivefollowerslist = ""
    relay_list = ["wss://relay.damus.io", "wss://nostr-pub.wellorder.net", "wss://nos.lol", "wss://nostr.wine",
                  "wss://relay.nostfiles.dev", "wss://nostr.mom", "wss://nostr.oxtr.dev", "wss://relay.nostr.bg", "wss://relay.f7z.io"]
    relaytimeout = 5
    step = 25
    keys = Keys.from_sk_str(dvmkey)
    opts = Options().wait_for_send(wait_for_send).send_timeout(timedelta(seconds=5)).skip_disconnected_relays(True)
    cl = Client.with_opts(keys, opts)

    for relay in relay_list:
        cl.add_relay(relay)
    cl.connect()

    filt = Filter().author(user).kind(3).limit(1)
    followers = cl.get_events_of([filt], timedelta(seconds=relaytimeout))
    cl.disconnect()

    if len(followers) > 0:


        resultlist = []
        newest = 0
        best_entry = followers[0]
        for entry in followers:
            if entry.created_at().as_secs() > newest:
                newest = entry.created_at().as_secs()
                best_entry = entry

        print(best_entry.as_json())
        followings = []
        ns.dic = {}
        for tag in best_entry.tags():
            if tag.as_vec()[0] == "p":
                following = tag.as_vec()[1]
                followings.append(following)
                ns.dic[following] = "False"
        print("Followings: " + str(len(followings)))

        notactivesinceseconds = int(notactivesincedays) * 24 * 60 * 60
        dif = Timestamp.now().as_secs() - notactivesinceseconds
        notactivesince = Timestamp.from_secs(dif)

        def scanList(followings, instance, i, st, notactivesince):
            from nostr_sdk import Filter
            relay_list = ["wss://relay.damus.io", "wss://nostr-pub.wellorder.net", "wss://nos.lol", "wss://nostr.wine",
                          "wss://relay.nostfiles.dev", "wss://nostr.mom", "wss://nostr.oxtr.dev", "wss://relay.nostr.bg", "wss://relay.f7z.io"]
            keys = Keys.from_sk_str(dvmkey)
            opts = Options().wait_for_send(wait_for_send).send_timeout(timedelta(seconds=5)).skip_disconnected_relays(True)
            cli = Client.with_opts(keys, opts)
            for relay in relay_list:
                cli.add_relay(relay)
            cli.connect()

            filters = []
            for i in range(i + st):
                filter1 = Filter().author(followings[i]).since(notactivesince).limit(1)
                filters.append(filter1)

            authors = cli.get_events_of(filters, timedelta(seconds=8))

            for author in authors:
                instance.dic[author.pubkey().to_hex()] = "True"

            print(str(i) + "/" + str(len(followings)))
            cli.disconnect()

        threads = []


        i = 0
        while i < len(followings) - step:
            args = [followings, ns, i, step, notactivesince]
            t = Thread(target=scanList, args=args)
            threads.append(t)
            i = i+step

        #last to step size
        missing_scans = (len(followings) - i)
        args = [followings, ns, i, missing_scans, notactivesince]
        t = Thread(target=scanList, args=args)
        threads.append(t)

        # Start all threads
        for x in threads:
            x.start()

        # Wait for all of them to finish
        for x in threads:
            x.join()

        filters = []
        result = {k for (k, v) in ns.dic.items() if v == "False"}

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


