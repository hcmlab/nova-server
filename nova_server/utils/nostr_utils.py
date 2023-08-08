import json

import os
import re
import urllib
from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import urlparse

import nostr_sdk.nostr_sdk
import requests
import emoji
import ffmpegio
from decord import AudioReader, cpu
from nostr_sdk import Keys, Client, Tag, Event, EventBuilder, Filter, HandleNotification, Timestamp, nip04_decrypt, EventId, init_logger, LogLevel
import time
from nostr_sdk.nostr_sdk import PublicKey

from nova_server.utils.db_utils import db_entry_exists, add_new_session_to_db
from nova_server.utils.mediasource_utils import download_podcast, downloadYouTube, checkYoutubeLinkValid
from configparser import ConfigParser

# TODO
# check expiry of tasks/available output format/model/ (task is checked already). if not available ignore the job,
# send reaction on error (send sats back ideally, find out with lib, same for under payment),
# send reaction processing-scheduled when task is waiting for previous task to finish, max limit to wait?
# store whitelist (and maybe a blacklist) in a config/db
# clear list of  tasks (JobstoWatch) to watch after some time (timeout if invoice not paid),
# consider max-sat amount at all,
# consider reactions from customers (Kind 65000 event)
# add more output formats (webvtt, srt)
# add summarization task (GPT4all?, OpenAI api?) in own module
# purge database and files from time to time?
# Show preview of longer transcriptions, then ask for zap
# TASK: OCR
# TASK: LLAMA2
# TASK:


class DVMConfig:
    SUPPORTED_TASKS = ["speech-to-text", "translation", "text-to-image", "image-to-image", "image-upscale", "chat", "ocr"]
    LNBITS_INVOICE_KEY = 'bfdfb5ecfc0743daa08749ce58abea74'
    LNBITS_INVOICE_URL = 'https://ln.novaannotation.com/createLightningInvoice'
    AUTOPROCESS_MIN_AMOUNT: int = 1000000000000  # auto start processing if min Sat amount is given
    AUTOPROCESS_MAX_AMOUNT: int = 0  # if this is 0 and min is very big, autoprocess will not trigger
    SHOWRESULTBEFOREPAYMENT: bool = True  # if this flag is true show results even when not paid (in the end, right after autoprocess)
    COSTPERUNIT_TRANSLATION: int = 2  # Still need to multiply this by duration
    COSTPERUNIT_SPEECHTOTEXT: int = 5  # Still need to multiply this by duration
    COSTPERUNIT_IMAGEPROCESSING: int = 50  # Generate / Transform one image
    COSTPERUNIT_IMAGEUPSCALING: int = 500  # This takes quite long..


@dataclass
class JobToWatch:
    id: str
    timestamp: str
    isPaid: bool
    amount: int
    status: str
    result: str
    isProcessed: bool


JobstoWatch = []
relay_list = ["wss://relay.damus.io", "wss://blastr.f7z.xyz", "wss://relayable.org", "wss://nostr-pub.wellorder.net"]
relaytimeout = 1

#init_logger(LogLevel.DEBUG)

def nostr_client():
    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    sk = keys.secret_key()
    pk = keys.public_key()
    print(f"Nostr Bot public key: {pk.to_bech32()}")
    client = Client(keys)
    for relay in relay_list:
        client.add_relay(relay)
    client.connect()

    dmzapfilter = Filter().pubkey(pk).kinds([4, 9734, 9735]).since(Timestamp.now())
    dvmfilter = (Filter().kinds([66000, 65002, 65003, 65004, 65005]).since(Timestamp.now()))
    client.subscribe([dmzapfilter, dvmfilter])

    class NotificationHandler(HandleNotification):

        def handle(self, relay_url, event):
            print(f"Received new event from {relay_url}: {event.as_json()}")
            if (65002 <= event.kind() <= 66000):
                if isBlackListed(event.pubkey):
                    sendJobStatusReaction(event, "error")
                    print("Request by blacklisted user, skipped")
                elif checkTaskisSupported(event):
                    task = getTask(event)
                    if isWhiteListed(event.pubkey().to_hex(), task):
                        print("[Nostr] Whitelisted for task " + task + ". Starting processing..")
                        sendJobStatusReaction(event, "processing", True, 0)
                        doWork(event, isFromBot=False)
                    # otherwise send payment request
                    else:
                        bid = 0
                        for tag in event.tags():
                            if tag.as_vec()[0] == 'bid':
                                bid = int(tag.as_vec()[1])

                        amount = 10000
                        if task == "translation":
                            duration = 1  # todo get task duration
                            amount = DVMConfig.COSTPERUNIT_TRANSLATION * duration * 1000  # *1000 because millisats
                            print("[Nostr][Payment required] New Nostr translation Job event: " + event.as_json())
                        elif task == "speech-to-text":
                            duration = 1  # todo get task duration
                            amount = DVMConfig.COSTPERUNIT_TRANSLATION * duration * 1000  # *1000 because millisats
                            print("[Nostr][Payment required] New Nostr speech-to-text Job event: " + event.as_json())
                        elif task == "text-to-image":
                            amount = DVMConfig.COSTPERUNIT_IMAGEPROCESSING * 1000  # *1000 because millisats
                            print("[Nostr][Payment required] New Nostr generate Image Job event: " + event.as_json())
                        elif task == "image-to-image":
                            # todo get image size
                            amount = DVMConfig.COSTPERUNIT_IMAGEPROCESSING * 1000  # *1000 because millisats
                            print("[Nostr][Payment required] New Nostr convert Image Job event: " + event.as_json())
                        elif task == "image-upscale":
                            amount = DVMConfig.COSTPERUNIT_IMAGEUPSCALING * 1000  # *1000 because millisats
                            print("[Nostr][Payment required] New Nostr upscale Image Job event: " + event.as_json())
                        elif task == "chat":
                            amount = DVMConfig.COSTPERUNIT_IMAGEUPSCALING * 1000  # *1000 because millisats
                            print("[Nostr][Payment required] New Nostr Chat Job event: " + event.as_json())
                        elif task == "ocr":
                            amount = DVMConfig.COSTPERUNIT_TRANSLATION * 1000  # *1000 because millisats
                            print("[Nostr][Payment required] New Nostr OCR Job event: " + event.as_json())
                        else:
                            print("[Nostr] Task " + task + " is currently not supported by this instance")
                        if bid > 0:
                            willingtopay = bid
                            if willingtopay > DVMConfig.AUTOPROCESS_MIN_AMOUNT * 1000 or willingtopay < DVMConfig.AUTOPROCESS_MAX_AMOUNT * 1000:
                                print("[Nostr][Auto-processing: Payment suspended to end] Job event: " + str(
                                    event.as_json()))
                                doWork(event, isFromBot=False)
                            else:
                                if willingtopay >= amount:
                                    sendJobStatusReaction(event, "payment-required", False,
                                                          willingtopay)  # Take what user is willing to pay, min server rate
                                else:
                                    sendJobStatusReaction(event, "payment-rejected", False,
                                                          amount)  # Reject and tell user minimum amount

                        else:  # If there is no bid, just request server rate from user
                            print("[Nostr] Requesting payment for Event: " + event.id().to_hex())
                            sendJobStatusReaction(event, "payment-required", False, amount)
                else:
                    print("Got new Task but can't process it, skipping..")

            elif event.kind() == 4:
                try:
                    dec_text = nip04_decrypt(sk, event.pubkey(), event.content())
                    print(f"Received new msg: {dec_text}")

                    if str(dec_text).startswith("-text-to-image") or str(dec_text).startswith("-image-to-image") or str(dec_text).startswith("-speech-to-text") or str(dec_text).startswith("-image-upscale"):
                        task = str(dec_text).split(' ')[0].removeprefix('-')
                        time.sleep(3.0)
                        if isWhiteListed(event.pubkey().to_hex(), task):
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(),
                                                                          "Whitelisted, processing started.\n\nI will DM you once your task is ready.",
                                                                          None).to_event(keys)
                            sendEvent(evt)
                            tags = parsebotcommandtoevent(dec_text)
                            tags.append(Tag.parse(["p", event.pubkey().to_hex()]))
                            evt = EventBuilder(4, "", tags).to_event(keys)
                            print(evt.as_json())
                            doWork(evt, isFromBot=True)

                        else:
                             event = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(), "Payment required, please zap this note with at least " + str(DVMConfig.COSTPERUNIT_IMAGEPROCESSING) + " Sats ..", event.id()).to_event(keys)
                             JobstoWatch.append(JobToWatch(id=event.id().to_hex(), timestamp=event.created_at().as_secs(), amount=50,isPaid=False, status="payment-required", result="", isProcessed=False))
                             client.send_event(event)
                except Exception as e:
                    print(f"Error during content decryption: {e}")
            elif event.kind() == 9734:
                print(event.as_json())

            elif event.kind() == 9735:
                print("Zap received")
                try:
                    for tag in event.tags():
                        if tag.as_vec()[0] == 'bolt11':
                            invoicesats = ParseBolt11Invoice(tag.as_vec()[1])
                        elif tag.as_vec()[0] == 'e':
                            print(tag.as_vec()[1])
                            zapableevent = getEvent(tag.as_vec()[1])
                            if (zapableevent.kind() == 65000):  # if a reaction by us got zapped
                                for tag in zapableevent.tags():
                                    amount = 0
                                    if tag.as_vec()[0] == 'amount':
                                        amount = int(float(tag.as_vec()[1]))
                                    elif tag.as_vec()[0] == 'e':
                                        jobevent = getEvent(tag.as_vec()[1])
                                        print("[Nostr] Original Job Request event found...")

                                if amount <= invoicesats * 1000:
                                    print("[Nostr] Payment-request fulfilled...")
                                    sendJobStatusReaction(jobevent, "processing")
                                    indices = [i for i, x in enumerate(JobstoWatch) if x.id == jobevent.id().to_hex()]
                                    index = -1
                                    if len(indices) > 0:
                                        index = indices[0]
                                    if (index > -1):
                                        # todo also remove ids after x time of waiting, need to store pairs of id / timestamp for that
                                        if (JobstoWatch[index]).isProcessed:  # If payment-required appears after processing
                                            JobstoWatch[index].isPaid = True
                                            CheckEventStatus(JobstoWatch[index].result, str(jobevent.as_json()))
                                        elif not (JobstoWatch[index]).isProcessed:  # If payment-required appears before processing
                                            JobstoWatch.pop(index)
                                            doWork(jobevent, isFromBot=False)
                                else:
                                    sendJobStatusReaction(jobevent, "payment-rejected", invoicesats * 1000)
                                    print("[Nostr] Invoice was not paid sufficiently")

                            elif zapableevent.kind() == 4:
                                if invoicesats >= DVMConfig.COSTPERUNIT_IMAGEPROCESSING:
                                    print("[Nostr] Original Prompt Job Request event found...")
                                    for tag in zapableevent.tags():
                                        if tag.as_vec()[0] == 'e':
                                            evt = getEvent(tag.as_vec()[1])
                                            if evt == None:
                                                filter = Filter().id(tag.as_vec()[1]).limit(1)
                                                events = client.get_events_of([filter], timedelta(seconds=5))
                                                evt = events[0]
                                            print(evt.as_json())
                                            indices = [i for i, x in enumerate(JobstoWatch) if x.id == zapableevent.id().to_hex()]
                                            if len(indices) == 1:
                                                event = EventBuilder.new_encrypted_direct_msg(keys, evt.pubkey(), "Payment received, processing started.\n\nI will DM you once your task is ready.", None).to_event(keys)
                                                dec_text = nip04_decrypt(sk, evt.pubkey(), evt.content())
                                                JobstoWatch.pop(indices[0])
                                                print(JobstoWatch)
                                                sendEvent(event)

                                                tags = parsebotcommandtoevent(dec_text)
                                                tags.append(Tag.parse(["p", evt.pubkey().to_hex()]))
                                                event = EventBuilder(4, "", tags).to_event(keys)
                                                print(event.as_json())
                                                doWork(event, isFromBot=True)


                                            break
                            else:
                                print("[Nostr] Zap was not for a kind 65000 or 4 reaction, skipping")

                except Exception as e:
                    print(f"Error during content decryption: {e}")

        def handle_msg(self, relay_url, msg):
            None

    def createRequestFormfromNostrEvent(event, isBot=False):
        # Only call this if config is not available, adjust function to your db
        # savConfig()
        task = getTask(event)

        # Read config.ini file
        config_object = ConfigParser()
        config_object.read("nostrconfig.ini")
        if len(config_object) == 1:
            dbUser = input("Please enter a DB User:\n")
            dbPassword = input("Please enter DB User Password:\n")
            dbServer = input("Please enter a DB Host:\n")
            SaveConfig(dbUser, dbPassword, dbServer, "nostr_test", "nostr", "system")
            config_object.read("nostrconfig.ini")

        userinfo = config_object["USERINFO"]
        serverconfig = config_object["SERVERCONFIG"]

        request_form = {}
        request_form["dbServer"] = serverconfig["dbServer"]
        request_form["dbUser"] = userinfo["dbUser"]
        request_form["dbPassword"] = userinfo["dbPassword"]
        request_form["database"] = serverconfig["database"]
        request_form["roles"] = serverconfig["roles"]
        request_form["annotator"] = serverconfig["annotator"]
        request_form["flattenSamples"] = "false"
        request_form["jobID"] = event.id().to_hex()

        request_form["frameSize"] = 0
        request_form["stride"] = request_form["frameSize"]
        request_form["leftContext"] = 0
        request_form["rightContext"] = 0
        request_form["nostrEvent"] = event.as_json()
        request_form["sessions"] = event.id().to_hex()

        request_form["isBot"] = str(isBot)

        # defaults might be overwritten by nostr event

        request_form["startTime"] = "0"
        request_form["endTime"] = "0"

        if task == "speech-to-text":
            # Declare specific model type e.g. whisperx_large-v2
            request_form["mode"] = "PREDICT"
            alignment = "raw"
            modelopt = "large-v2"

            for tag in event.tags():
                if tag.as_vec()[0] == 'param':
                    print(tag.as_vec())
                    param = tag.as_vec()[1]
                    if param == "range":  # check for paramtype
                        try:
                            t = time.strptime(tag.as_vec()[2], "%H:%M:%S")
                            seconds = t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec
                            request_form["startTime"] = str(seconds)
                        except:
                            try:
                                t = time.strptime(tag.as_vec()[2], "%M:%S")
                                seconds = t.tm_min * 60 + t.tm_sec
                                request_form["startTime"] = str(seconds)
                            except:
                                request_form["startTime"] = tag.as_vec()[2]
                        try:
                            t = time.strptime(tag.as_vec()[3], "%H:%M:%S")
                            seconds = t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec
                            request_form["endTime"] = str(seconds)
                        except:
                            try:
                                t = time.strptime(tag.as_vec()[3], "%M:%S")
                                seconds = t.tm_min * 60 + t.tm_sec
                                request_form["endTime"] = str(seconds)
                            except:
                                request_form["endTime"] = tag.as_vec()[3]

                    elif param == "alignment":  # check for paramtype
                        alignment = tag.as_vec()[2]
                    elif param == "model":  # check for paramtype
                        modelopt = tag.as_vec()[2]



            request_form["schemeType"] = "FREE"
            request_form["scheme"] = "transcript"
            request_form["streamName"] = "audio"
            request_form["trainerFilePath"] = 'models\\trainer\\' + str(
            request_form["schemeType"]).lower() + '\\' + str(
            request_form["scheme"]) + '\\audio{audio}\\whisperx\\whisperx_transcript.trainer'
            request_form["optStr"] = 'model=' + modelopt + ';alignment_mode=' + alignment + ';batch_size=2'

        elif task == "translation":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'translation'
            # outsource this to its own script, ideally. This is not using the database for now, but probably should.
            inputtype = "event"
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    inputtype = tag.as_vec()[2]

                elif tag.as_vec()[0] == 'param':
                    param = tag.as_vec()[1]
                    if param == "language":  # check for paramtype
                        translation_lang = str(tag.as_vec()[2]).split('-')[0]

            if inputtype == "event":
                for tag in event.tags():
                    if tag.as_vec()[0] == 'i':
                        evt = getEvent(tag.as_vec()[1])
                        text = evt.content()
                        break

            elif inputtype == "text":
                for tag in event.tags():
                    if tag.as_vec()[0] == 'i':
                        text = tag.as_vec()[1]
                        break

            elif inputtype == "job":
                for tag in event.tags():
                    if tag.as_vec()[0] == 'i':
                        jobidfilter = Filter().kind(65001).event(EventId.from_hex(tag.as_vec()[1])).limit(1)
                        print(jobidfilter.as_json())
                        events = client.get_events_of([jobidfilter], timedelta(seconds=relaytimeout))
                        evt = events[0]
                        text = evt.content()
                        break

            request_form["optStr"] = 'text=' + text + ';translation_lang=' + translation_lang

        elif task == "ocr":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'ocr'
            # outsource this to its own script, ideally. This is not using the database for now, but probably should.
            inputtype = "url"
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    inputtype = tag.as_vec()[2]

                #elif tag.as_vec()[0] == 'param':
                #    param = tag.as_vec()[1]
                #    if param == "language":  # check for paramtype
                #        translation_lang = str(tag.as_vec()[2]).split('-')[0]


            if inputtype == "url":
                for tag in event.tags():
                    if tag.as_vec()[0] == 'i':
                        url = tag.as_vec()[1]
                        break

            request_form["optStr"] = 'url=' + url

        elif task == "image-to-image":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'image-to-image'
            prompt = ""
            negative_prompt = ""
            strength = 0.5
            guidance_scale = 7.5
            model = "timbrooks/instruct-pix2pix"

            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    inputtype = tag.as_vec()[2]
                    if inputtype == "url":
                        url = tag.as_vec()[1]
                    elif inputtype == "event":
                        evt = getEvent(tag.as_vec()[1])
                        url = re.search("(?P<url>https?://[^\s]+)", evt.content()).group("url")
                    elif inputtype == "job":
                        jobidfilter = Filter().kind(65001).event(EventId.from_hex(tag.as_vec()[1])).limit(1)
                        print(jobidfilter.as_json())
                        events = client.get_events_of([jobidfilter], timedelta(seconds=relaytimeout))
                        evt = events[0]
                        url = evt.content()
                elif tag.as_vec()[0] == 'param':
                    if tag.as_vec()[1] == "prompt":  # check for paramtype
                        prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "negative_prompt":  # check for paramtype
                        negative_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "strength":  # check for paramtype
                        strength = float(tag.as_vec()[2])
                    elif tag.as_vec()[1] == "guidance_scale":  # check for paramtype
                        guidance_scale = float(tag.as_vec()[2])
                    elif tag.as_vec()[1] == "model":  # check for paramtype
                        if str(tag.as_vec()[2]).lower().__contains__("gta"):
                            model = "GTA5_Artwork_Diffusion_gtav_style"
                        elif str(tag.as_vec()[2]).lower().__contains__("realistic"):
                            model = "realisticVisionV51_v51VAE"
                        elif str(tag.as_vec()[2]).lower().__contains__("sdxl"):
                            model = "stabilityai/stable-diffusion-xl-refiner-1.0"
                        elif str(tag.as_vec()[2]).lower().__contains__("wild"):
                            model = "stablydiffusedsWild_351"
                        elif str(tag.as_vec()[2]).lower().__contains__("pix2pix"):
                            model = "timbrooks/instruct-pix2pix"
                        else:
                            model = "timbrooks/instruct-pix2pix"

            request_form["optStr"] = 'url=' + url + ';prompt=' + prompt + ';negative_prompt=' + negative_prompt + ';strength=' + str(strength) + ';guidance_scale=' + str(guidance_scale) + ';model=' + model

        elif task == "text-to-image":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'text-to-image'
            width = "768"
            height = "768"
            extra_prompt = ""
            negative_prompt = ""
            upscale = "4"
            model = "stabilityai/stable-diffusion-xl-base-1.0"

            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    type = tag.as_vec()[2]
                    if type == "text":
                        prompt = tag.as_vec()[1]
                    elif type == "event":
                        evt = getEvent(tag.as_vec()[1])
                        prompt = evt.content()
                    elif type == "job":
                        jobidfilter = Filter().kind(65001).event(EventId.from_hex(tag.as_vec()[1])).limit(1)
                        events = client.get_events_of([jobidfilter], timedelta(seconds=relaytimeout))
                        evt = events[0]
                        prompt = evt.content()
                elif tag.as_vec()[0] == 'param':
                    if tag.as_vec()[1] == "prompt":  # check for paramtype
                        extra_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "negative_prompt":  # check for paramtype
                        negative_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "size":  # check for paramtype
                        width = tag.as_vec()[2]
                        height = tag.as_vec()[3]
                    elif tag.as_vec()[1] == "upscale":  # check for paramtype
                        upscale = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "model":  # check for paramtype
                        if str(tag.as_vec()[2]).lower().__contains__("gta"):
                            model  = "GTA5_Artwork_Diffusion_gtav_style"
                        elif str(tag.as_vec()[2]).lower().__contains__("realistic"):
                            model = "realisticVisionV51_v51VAE"
                        elif str(tag.as_vec()[2]).lower().__contains__("sdxl"):
                            model = "stabilityai/stable-diffusion-xl-base-1.0"
                        elif str(tag.as_vec()[2]).lower().__contains__("sd15"):
                            model = "runwayml/stable-diffusion-v1-5"
                        elif str(tag.as_vec()[2]).lower().__contains__("wild"):
                            model = "stablydiffusedsWild_351"
                        elif str(tag.as_vec()[2]).lower().__contains__("lora"):
                            model = str(tag.as_vec()[2])
                        else:
                            model = "stabilityai/stable-diffusion-xl-base-1.0"

            request_form["optStr"] = 'prompt=' + prompt + ';extra_prompt=' + extra_prompt + ';negative_prompt=' + negative_prompt + ';width=' + width + ';height=' + height + ';upscale=' + upscale + ';model=' + model

        elif task == "image-upscale":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'image-upscale'
            upscale = "4"
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    inputtype = tag.as_vec()[2]
                    if inputtype == "url":
                        url = tag.as_vec()[1]
                    elif inputtype == "event":
                        evt = getEvent(tag.as_vec()[1])
                        url = re.search("(?P<url>https?://[^\s]+)",  evt.content()).group("url")
                    elif inputtype == "job":
                        jobidfilter = Filter().kind(65001).event(EventId.from_hex(tag.as_vec()[1])).limit(1)
                        print(jobidfilter.as_json())
                        events = client.get_events_of([jobidfilter], timedelta(seconds=relaytimeout))
                        evt = events[0]
                        url = evt.content()
                elif tag.as_vec()[0] == 'param':
                    if tag.as_vec()[1] == "upscale":  # check for paramtype
                        upscale = tag.as_vec()[2]

            request_form["optStr"] = 'url=' + url + ";upscale=" + upscale

        elif task == "chat":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'chat'
            print("[Nostr] Chat request ")
            text = ""
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    text = tag.as_vec()[1]
            request_form["optStr"] = 'message=' + text

            # add length variableF
        elif task == "summarization":
            request_form["mode"] = "PREDICT_STATIC"
            print("[Nostr] Not supported yet")
            # call OpenAI API or use a local LLM
            # add length variableF

        return request_form
    def doWork(Jobevent, isFromBot=False):
        if (Jobevent.kind() >= 65002 and Jobevent.kind() <= 66000) or Jobevent.kind() == 68001 or Jobevent.kind() == 4:
            request_form = createRequestFormfromNostrEvent(Jobevent, isFromBot)
            task = getTask(Jobevent)
            if task == "speech-to-text":
                print("[Nostr] Adding Nostr speech-to-text Job event: " + Jobevent.as_json())
                organizeInputData(Jobevent, request_form)
            elif task == "event-list-generation" or task == "summarization" or task == "chat" or task.startswith(
                    "unknown"):
                print("Task not (yet) supported")
                return
            else:
                print("[Nostr] Adding " + task + " Job event: " + Jobevent.as_json())

            url = 'http://' + os.environ["NOVA_HOST"] + ':' + os.environ["NOVA_PORT"] + '/' + str(request_form["mode"]).lower()
            headers = {'Content-type': 'application/x-www-form-urlencoded'}
            requests.post(url, headers=headers, data=request_form)


    client.handle_notifications(NotificationHandler())
    while True:
        time.sleep(5.0)

def getEvent(eventidstr):
    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    cl = Client(keys)
    for relay in relay_list:
        cl.add_relay(relay)
    cl.connect()
    filter = Filter().id(eventidstr).limit(1)
    events = cl.get_events_of([filter], timedelta(seconds=relaytimeout))
    cl.disconnect()
    if len(events) > 0:
        return events[0]
    else:
        return None
def sendEvent(event):
    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    cl = Client(keys)

    for tag in event.tags():
        if tag.as_vec()[0] == 'relays':
            relays = tag.as_vec()[1].split(',')
            for relay in relays:
                cl.add_relay(relay)
    else:
        for relay in relay_list:
            cl.add_relay(relay)

    cl.connect()
    id = cl.send_event(event)
    cl.disconnect()
    return id
def organizeInputData(event, request_form):
    data_dir = os.environ["NOVA_DATA_DIR"]

    session = event.id().to_hex()
    inputtype = "url"
    for tag in event.tags():
        if tag.as_vec()[0] == 'i':
            input = tag.as_vec()[1]
            inputtype = tag.as_vec()[2]
            break

    if inputtype == "url":
        if not os.path.exists(data_dir + '\\' + request_form["database"] + '\\' + session):
            os.mkdir(data_dir + '\\' + request_form["database"] + '\\' + session)
        # We can support some services that don't use default media links, like overcastfm for podcasts
        if str(input).startswith("https://overcast.fm/"):
            filename = data_dir + '\\' + request_form["database"] + '\\' + session + '\\' + request_form[
                "roles"] + ".originalaudio.mp3"
            print("Found overcast.fm Link.. downloading")
            download_podcast(input, filename)
            finaltag = str(input).replace("https://overcast.fm/", "").split('/')
            if float(request_form["startTime"]) == 0.0:
                if (len(finaltag) > 1):
                    t = time.strptime(finaltag[1], "%H:%M:%S")
                    seconds = t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec
                    request_form["startTime"] = str(seconds)  # overwrite from link.. why not..
                    print("Setting start time automatically to " + request_form["startTime"])
                    if float(request_form["endTime"]) > 0.0:
                        request_form["endTime"] = seconds + float(request_form["endTime"])
                        print("Moving end time automatically to " + request_form["endTime"])

        # is youtube link?
        elif str(input).replace("http://", "").replace("https://", "").replace("www.", "").replace("youtu.be/",
                                                                                                   "youtube.com?v=")[
             0:11] == "youtube.com":

            filepath = data_dir + '\\' + request_form["database"] + '\\' + session + '\\'
            try:
                filename = downloadYouTube(input, filepath)
                o = urlparse(input)
                q = urllib.parse.parse_qs(o.query)

                if float(request_form["startTime"]) == 0.0:
                    if (o.query.find('t=') != -1):
                        request_form["startTime"] = q['t'][0]  # overwrite from link.. why not..
                        print("Setting start time automatically to " + request_form["startTime"])
                        if float(request_form["endTime"]) > 0.0:
                            request_form["endTime"] = str(float(q['t'][0]) + float(request_form["endTime"]))
                            print("Moving end time automatically to " + request_form["endTime"])
            except Exception:
                print("video not available")
                sendJobStatusReaction(event, "error")
                return
        # Regular links have a media file ending and/or mime types
        else:
            req = requests.get(input)
            content_type = req.headers['content-type']
            if content_type == 'audio/x-wav' or str(input).endswith(".wav"):
                ext = "wav"
                type = "audio"
            elif content_type == 'audio/mpeg' or str(input).endswith(".mp3"):
                ext = "mp3"
                type = "audio"
            elif content_type == 'audio/ogg' or str(input).endswith(".ogg"):
                ext = "ogg"
                type = "audio"
            elif content_type == 'video/mp4' or str(input).endswith(".mp4"):
                ext = "mp4"
                type = "video"
            elif content_type == 'video/avi' or str(input).endswith(".avi"):
                ext = "avi"
                type = "video"
            elif content_type == 'video/mov' or str(input).endswith(".mov"):
                ext = "mov"
                type = "video"

            else:
                sendJobStatusReaction(event, "error")
                return

            filename = data_dir + '\\' + request_form["database"] + '\\' + session + '\\' + request_form[
                "roles"] + '.original' + type + '.' + ext

            if not os.path.exists(filename):
                file = open(filename, 'wb')
                for chunk in req.iter_content(100000):
                    file.write(chunk)
                file.close()

        duration = 0
        try:
            file_reader = AudioReader(filename, ctx=cpu(0), mono=False)
            duration = file_reader.duration()
        except:
            sendJobStatusReaction(event, "error")
            return

        print("Duration of the Media file: " + str(duration))
        if float(request_form['endTime']) == 0.0:
            end_time = float(duration)
        elif float(request_form['endTime']) > duration:
            end_time =  float(duration)
        else:
            end_time = float(request_form['endTime'])
        if (float(request_form['startTime']) < 0.0 or float(request_form['startTime']) > end_time):
            start_time = 0.0
        else:
            start_time = float(request_form['startTime'])

        print("Converting from " + str(start_time) + " until " + str(end_time))
        # for now we cut and convert all files to mp3
        finalfilename = data_dir + '\\' + request_form["database"] + '\\' + session + '\\' + request_form[
            "roles"] + '.' + request_form["streamName"] + '.mp3'
        fs, x = ffmpegio.audio.read(filename, ss=start_time, to=end_time, sample_fmt='dbl', ac=1)
        ffmpegio.audio.write(finalfilename, fs, x)

        if not db_entry_exists(request_form, session, "name", "Sessions"):
            duration = end_time - start_time
            add_new_session_to_db(request_form, duration)
def isBlackListed(pubkey):
    # Store  lists of blacklisted npubs that can no do processing
    # todo blacklisting and whitelsting should be moved to a database and probably get some expiry
    blacklisted_all_tasks = []
    if any(pubkey == c for c in blacklisted_all_tasks):
        return True
    return False
def isWhiteListed(pubkey, task):
    # Store  ists of whistlisted npubs that can do free processing for specific tasks

    localnostrtest = "558497db304332004e59387bc3ba1df5738eac395b0e56b45bfb2eb5400a1e39"
    dbth = '99bb5591c9116600f845107d31f9b59e2f7c7e09a1ff802e84f1d43da557ca64'
    pablof7z = "fa984bd7dbb282f07e16e7ae87b26a2a7b9b90b7246a44771f0cf5ae58018f52"
    # PublicKey.from_npub("npub...").hex()
    whitelsited_npubs_speechtotext = []  # remove this to test LN Zaps
    whitelsited_npubs_translation = [localnostrtest]
    whitelsited_npubs_texttoimage = [localnostrtest]
    whitelsited_npubs_imagetoimage = [localnostrtest]
    whitelsited_npubs_imageupscale = [localnostrtest]
    whitelsited_npubs_chat = [localnostrtest]
    whitelsited_npubs_ocr = [localnostrtest]

    whitelsited_all_tasks = [dbth, localnostrtest]

    if (task == "speech-to-text"):
        if any(pubkey == c for c in whitelsited_npubs_speechtotext) or any(
                pubkey == c for c in whitelsited_all_tasks):
            return True
    elif (task == "translation"):
        if any(pubkey == c for c in whitelsited_npubs_translation) or any(
                pubkey == c for c in whitelsited_all_tasks):
            return True
    elif (task == "text-to-image"):
        if any(pubkey == c for c in whitelsited_npubs_texttoimage) or any(
                pubkey == c for c in whitelsited_all_tasks):
            return True
    elif (task == "image-to-image"):
        if any(pubkey == c for c in whitelsited_npubs_imagetoimage) or any(
                pubkey == c for c in whitelsited_all_tasks):
            return True
    elif (task == "image-upscale"):
        if any(pubkey == c for c in whitelsited_npubs_imageupscale) or any(
                pubkey == c for c in whitelsited_all_tasks):
            return True
    elif (task == "chat"):
        if any(pubkey == c for c in whitelsited_npubs_chat) or any(pubkey == c for c in whitelsited_all_tasks):
            return True
    elif (task == "ocr"):
        if any(pubkey == c for c in whitelsited_npubs_ocr) or any(pubkey == c for c in whitelsited_all_tasks):
            return True
    return False
def getTask(event):
        if event.kind() == 66000:  # use this for events that have no id yet
            for tag in event.tags():
                if tag.as_vec()[0] == 'j':
                    return tag.as_vec()[1]
            else:
                return "unknown job: " + event.as_json()
        elif event.kind() == 4:  # dm
            for tag in event.tags():
                if tag.as_vec()[0] == 'j':
                    return tag.as_vec()[1]
            else:
                return "unknown job: " + event.as_json()
        elif event.kind() == 65002:
            for tag in event.tags():
                if tag.as_vec()[0] == "i":
                    if tag.as_vec()[2] == "url":
                       type = CheckUrlisReadable(tag.as_vec()[1])
                       if type == "audio" or type == "video":
                           return "speech-to-text"
                       elif type == "image":
                            return "ocr"
                       else:
                           return "unknown job"
        elif event.kind() == 65003:
            return "summarization"
        elif event.kind() == 65004:
            return "translation"
        elif event.kind() == 65005:
            return "text-to-image"
        elif event.kind() == 65006:
            return "event-list-generation"
        else:
            return "unknown type"
def CheckEventStatus(content, originaleventstr: str, useBot=False):
    originalevent = Event.from_json(originaleventstr)
    for x in JobstoWatch:
        if x.id == originalevent.id().to_hex():
            isPaid = x.isPaid
            amount = x.amount
            x.result = content
            x.isProcessed = True
            if DVMConfig.SHOWRESULTBEFOREPAYMENT and not isPaid:
                sendNostrReplyEvent(content, originaleventstr)
                sendJobStatusReaction(originalevent, "success", amount)  # or payment-required, or both?
            elif not DVMConfig.SHOWRESULTBEFOREPAYMENT and not isPaid:
                sendJobStatusReaction(originalevent, "success", amount)  # or payment-required, or both?

            if (DVMConfig.SHOWRESULTBEFOREPAYMENT and isPaid):
                JobstoWatch.remove(x)
            elif not DVMConfig.SHOWRESULTBEFOREPAYMENT and isPaid:
                JobstoWatch.remove(x)
                sendNostrReplyEvent(content, originaleventstr)
            print(str(JobstoWatch))
            break

    else:
        resultcontent = postprocessResult(content, originalevent)
        print(str(JobstoWatch))
        if (useBot):
            keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
            for tag in originalevent.tags():
                if tag.as_vec()[0] == "p":
                    reckey = PublicKey.from_hex(tag.as_vec()[1])
            event = EventBuilder.new_encrypted_direct_msg(keys, reckey, "Your Result: \n\n" + resultcontent, None).to_event(keys)
            sendEvent(event)

        else:
            sendNostrReplyEvent(resultcontent, originaleventstr)
            sendJobStatusReaction(originalevent, "success")
def sendNostrReplyEvent(content, originaleventstr):
    originalevent = Event.from_json(originaleventstr)
    requesttag = Tag.parse(["request", originaleventstr.replace("\\", "")])
    etag = Tag.parse(["e", originalevent.id().to_hex()])
    ptag = Tag.parse(["p", originalevent.pubkey().to_hex()])
    alttag = Tag.parse(["alt", "This is the result of a NIP90 DVM AI task with kind " + str(originalevent.kind()) + ". The task was: "+ originalevent.content()])
    statustag = Tag.parse(["status", "success"])

    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    event = EventBuilder(65001, str(content), [requesttag, etag, ptag, alttag, statustag]).to_event(keys)
    sendEvent(event)
    print("[Nostr] 65001 Job Response event sent: " + event.as_json())
    return event.as_json()
def sendJobStatusReaction(originalevent, status, isPaid=True, amount=0):
        altdesc = "This is a reaction to a NIP90 DVM AI task."
        task = getTask(originalevent)
        if status == "processing":
            reaction = emoji.emojize(":thumbs_up:")
            altdesc =   "NIP90 DVM AI task " + task + " started processing."
        elif status == "success":
            reaction = emoji.emojize(":call_me_hand:")
            altdesc = "NIP90 DVM AI task " + task + " finished successfully."
        elif status == "error":
            reaction = emoji.emojize(":thumbs_down:")
            altdesc = "NIP90 DVM AI task " + task + " had an error. So sorry. In the future zaps will be sent back but I can't do that just yet."
        elif status == "payment-required":
            reaction = emoji.emojize(":orange_heart:")
            altdesc = "NIP90 DVM AI task " + task + " requires payment of min "+ int(amount/1000) + " Sats."
            if task == "speech-to-text":
                altdesc = altdesc + " Providing results with WhisperX large-v2. Accepted input formats: wav,mp3,mp4,ogg,avi,mov,youtube,overcast. Possible outputs: text/plain, timestamped labels depending on alignment parameter (word,segment,raw)"
            elif task == "ocr":
                altdesc = altdesc + " Accepted input formats: jpg. Possible outputs: text/plain. This is very experimental, make sure your text is well readable."

        elif status == "payment-rejected":
            reaction = emoji.emojize(":see_no_evil_monkey:")
            altdesc = "NIP90 DVM AI task " + task + " payment is below required amount of " + int(amount/1000) + " Sats."
        elif status == "user-blocked-from-service":
            reaction = emoji.emojize(":see_no_evil_monkey:")
            altdesc = "NIP90 DVM AI task " + task + " can't be performed. User has been blocked from Service"
        else:
            reaction = emoji.emojize(":see_no_evil_monkey:")


        etag = Tag.parse(["e", originalevent.id().to_hex()])
        ptag = Tag.parse(["p", originalevent.pubkey().to_hex()])
        alttag = Tag.parse(["alt", altdesc])
        statustag = Tag.parse(["status", status])
        tags = [etag, ptag, alttag, statustag]

        if status == "success" or status == "error":  #
            for x in JobstoWatch:
                if x.id == originalevent.id():
                    isPaid = x.isPaid
                    amount = x.amount
                    break
        if status == "payment-required" or (status == "processing" and not isPaid):
            JobstoWatch.append(
                JobToWatch(id=originalevent.id().to_hex(), timestamp=originalevent.created_at().as_secs(), amount=amount, isPaid=isPaid,
                           status=status, result="", isProcessed=False))
            print(str(JobstoWatch))
        if status == "payment-required" or status == "payment-rejected" or (status == "processing" and not isPaid) or (status == "success" and not isPaid):
            #try:
            #    if DVMConfig.LNBITS_INVOICE_KEY != "":
            #        bolt11 = createBolt11LnBits(amount)
            #        amounttag = Tag.parse(["amount", str(amount), bolt11])
            #    else:
            #        amounttag = Tag.parse(["amount", str(amount)])
            #    tags.append(amounttag)
            #except:
            #    amounttag = Tag.parse(["amount", str(amount)])
            #    tags.append(amounttag)
            #    print("Couldn't get bolt11 invoice")
            amounttag = Tag.parse(["amount", str(amount)])
            tags.append(amounttag)

        keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
        event = EventBuilder(65000, reaction, tags).to_event(keys)
        event_id = sendEvent(event)
        print("[Nostr] Sent Kind 65000 Reaction: " + status + " " + event.as_json())
        return event.as_json()
def postprocessResult(content, originalevent):
    for tag in originalevent.tags():
        if tag.as_vec()[0] == "output":
            if tag.as_vec()[1] == "text/plain":
                result = ""
                try:
                    for name in content["name"]:
                        clearedName = str(name).lstrip("\'").rstrip("\'")
                        result = result + clearedName + "\n"
                    content = str(result).replace("\"", "").replace('[', "").replace(']', "").lstrip(None)
                except:
                    print("Can't transform text, or text already in text/plain format.")
            # TODO add more

    return content
def parsebotcommandtoevent(dec_text):

    dec_text = dec_text.replace("\n", "")
    if str(dec_text).startswith("-text-to-image"):
        negative_prompt = ""
        prompttemp = dec_text.replace("-text-to-image ", "")
        split = prompttemp.split("-")
        prompt = split[0]
        width = "768"
        height = "768"
        jTag = Tag.parse(["j", "text-to-image"])
        iTag = Tag.parse(["i", prompt, "text"])
        tags = [jTag, iTag]
        if len(split) > 1:
            for i in split:
                if i.startswith("negative"):
                    negative_prompt = i.replace("negative ", "")
                    paramTag = Tag.parse(["param", "negative_prompt", negative_prompt])
                    tags.append(paramTag)
                elif i.startswith("extra"):
                    extra_prompt = i.replace("extra ", "")
                    paramTag = Tag.parse(["param", "prompt", extra_prompt])
                    tags.append(paramTag)
                elif i.startswith("upscale"):
                    upscale_factor = i.replace("upscale ", "")
                    paramTag = Tag.parse(["param", "upscale", upscale_factor])
                    tags.append(paramTag)
                elif i.startswith("model"):
                    model = i.replace("model ", "")
                    paramTag = Tag.parse(["param", "model", model])
                    tags.append(paramTag)
                elif i.startswith("width"):
                    width = i.replace("width ", "")
                elif i.startswith("height"):
                    height = i.replace("height ", "")

        paramSizeTag = Tag.parse(["param", "size", width, height])
        tags.append(paramSizeTag)

        return tags

    elif str(dec_text).startswith("-image-to-image"):
        negative_prompt = ""
        prompttemp = dec_text.replace("-image-to-image ", "")
        split = prompttemp.split("-")
        url = str(split[0]).replace(' ', '')
        width = "768"
        height = "768"
        jTag = Tag.parse(["j", "image-to-image"])
        iTag = Tag.parse(["i", url, "url"])
        tags = [jTag, iTag]
        if len(split) > 1:
            for i in split:
                if i.startswith("negative"):
                    negative_prompt = i.replace("negative ", "")
                    paramTag = Tag.parse(["param", "negative_prompt", negative_prompt])
                    tags.append(paramTag)
                elif i.startswith("prompt"):
                    extra_prompt = i.replace("prompt ", "")
                    paramTag = Tag.parse(["param", "prompt", extra_prompt])
                    tags.append(paramTag)
                elif i.startswith("strength"):
                    strength = i.replace("strength ", "")
                    paramTag = Tag.parse(["param", "strength", strength])
                    tags.append(paramTag)
                elif i.startswith("guidance_scale"):
                    strength = i.replace("guidance_scale ", "")
                    paramTag = Tag.parse(["param", "guidance_scale", strength])
                    tags.append(paramTag)
                elif i.startswith("model"):
                    model = i.replace("model ", "")
                    paramTag = Tag.parse(["param", "model", model])
                    tags.append(paramTag)

            paramSizeTag = Tag.parse(["param", "size", width, height])
            tags.append(paramSizeTag)

            return tags

    elif str(dec_text).startswith("-image-upscale"):
        prompttemp = dec_text.replace("-image-upscale ", "")
        split = prompttemp.split("-")
        url = split[0]
        jTag = Tag.parse(["j", "image-upscale"])
        iTag = Tag.parse(["i", url, "url"])
        tags = [jTag, iTag]
        if len(split) > 1:
            for i in split:
                if i.startswith("upscale"):
                    upscale_factor = i.replace("upscale ", "")
                    paramTag = Tag.parse(["param", "upscale", upscale_factor])
                    tags.append(paramTag)
        return tags

    elif str(dec_text).startswith("-speech-to-text"):
        prompttemp = dec_text.replace("-speech-to-text ", "")
        split = prompttemp.split("-")
        url = split[0]
        start = "0"
        end = "0"
        model = "large-v2"
        if len(split) > 1:
            for i in split:
                if i.startswith("from"):
                    start = i.replace("from ", "")
                elif i.startswith("to"):
                    end = i.replace("to ", "")
                elif i.startswith("model"):
                    model = i.replace("model ", "")
        jTag = Tag.parse(["j", "speech-to-text"])
        iTag = Tag.parse(["i", url, "url"])
        oTag = Tag.parse(["output", "text/plain"])
        paramTag1 = Tag.parse(["param", "model", model])
        paramTag = Tag.parse(["param", "range", start, end])
        return [jTag, iTag, oTag, paramTag1, paramTag]

    elif str(dec_text).startswith("-chat"):
        text = dec_text.replace("-chat ", "")
        jTag = Tag.parse(["j", "chat"])
        iTag = Tag.parse(["i", text, "text"])
        oTag = Tag.parse(["output", "text/plain"])
        return [jTag, iTag, oTag]
def checkTaskisSupported(event):
    task = getTask(event)
    print("Received new Task: " + task)
    hasitag = False
    for tag in event.tags():
        if tag.as_vec()[0] == 'i':
            print(tag.as_vec())
            input = tag.as_vec()[1]
            inputtype = tag.as_vec()[2]
            hasitag = True


        elif tag.as_vec()[0] == 'output':
            output = tag.as_vec()[1]
            if output != "text/plain":
                return False

    if not hasitag:
        return False

    if task not in DVMConfig.SUPPORTED_TASKS:  # The Tasks this DVM supports (can be extended)
        return False
    if task == "translation" and (inputtype != "event" and inputtype != "job" and inputtype != "text"):  # The input types per task
        return False
    #if task == "translation" and len(event.content) > 4999:  # Google Services have a limit of 5000 signs
    #    return False
    if task == "speech-to-text" and (inputtype != "event" and inputtype != "job" and inputtype != "url"): # The input types per task
        return False
    if task == "image-upscale" and (inputtype != "event" and inputtype != "job" and inputtype != "url"):
        return False
    if inputtype == 'url' and CheckUrlisReadable(input) is None:
        return False

    return True
def CheckUrlisReadable(url):
    if not str(url).startswith("http"):
        return None
    # If it's a YouTube oder Overcast link, we suppose we support it
    if str(url).replace("http://", "").replace("https://", "").replace("www.", "").replace("youtu.be/",
                                                                                           "youtube.com?v=")[
       0:11] == "youtube.com" and str(url).find("live") == -1:
        if (checkYoutubeLinkValid(url)):
            return "video"

    elif str(url).startswith("https://overcast.fm/"):
        return "audio"

    # If link is comaptible with one of these file formats, it's fine.
    req = requests.get(url)
    content_type = req.headers['content-type']
    if content_type == 'audio/x-wav' or str(url).endswith(".wav") or content_type == 'audio/mpeg' or str(url).endswith(".mp3") or content_type == 'audio/ogg' or str(url).endswith(".ogg"):
            return "audio"
    elif content_type == 'image/png' or str(url).endswith(".png") or content_type == 'image/jpg' or str(url).endswith(".jpg") or content_type == 'image/jpeg' or str(url).endswith(".jpeg") or content_type == 'image/png' or str(url).endswith(".png"):
            return "image"
    elif content_type == 'video/mp4' or str(url).endswith(".mp4")  or content_type == 'video/avi' or str(url).endswith(".avi")  or content_type == 'video/mov' or str(url).endswith(".mov"):
            return "video"
    # Otherwise we will not offer to do the job.
    return None
def SaveConfig(dbUser, dbPassword, dbServer, database, role, annotator):
    # Get the configparser object
    config_object = ConfigParser()

    # Assume we need 2 sections in the config file, let's call them USERINFO and SERVERCONFIG
    config_object["USERINFO"] = {
        "dbUser": dbUser,
        "dbPassword": dbPassword
    }

    config_object["SERVERCONFIG"] = {
        "dbServer": dbServer,
        "database": database,
        "roles": role,
        "annotator": annotator
    }

    # Write the above sections to config.ini file
    with open('nostrconfig.ini', 'w') as conf:
        config_object.write(conf)
def getIndexOfFirstLetter(ip):
    index = 0
    for c in ip:
        if c.isalpha():
            return index
        else:
            index = index + 1

    return len(input);
def ParseBolt11Invoice(invoice):
    remaininginvoice = invoice[4:]
    index = getIndexOfFirstLetter(remaininginvoice)
    identifier = remaininginvoice[index]
    numberstring = remaininginvoice[:index]
    number = float(numberstring)
    if (identifier == 'm'):
        number = number * 100000000 * 0.001
    elif (identifier == 'u'):
        number = number * 100000000 * 0.000001
    elif (identifier == 'n'):
        number = number * 100000000 * 0.000000001
    elif (identifier == 'p'):
        number = number * 100000000 * 0.000000000001

    return int(number)
def createBolt11LnBits(millisats):
    sats = int(millisats / 1000)
    url = DVMConfig.LNBITS_INVOICE_URL
    data = {}
    data['invoice_key'] = DVMConfig.LNBITS_INVOICE_KEY
    data['sats'] = str(sats)
    data['memo'] = "Nostr-DVM"
    res = requests.post(url, data=data)
    obj = json.loads(res.text)
    return obj["payment_request"]

if __name__ == '__main__':
    os.environ["NOVA_DATA_DIR"] = "W:\\nova\\data"
    os.environ["NOVA_NOSTR_KEY"] = "privkey"
    os.environ["NOVA_HOST"] = "127.0.ß.1"
    os.environ["NOVA_PORT"] = "27017"
    nostr_client()


