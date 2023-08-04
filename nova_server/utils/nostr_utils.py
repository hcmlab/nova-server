import os
import urllib
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlparse
import requests
import emoji
import re
import ffmpegio
from decord import AudioReader, cpu
from nostr_sdk import Keys, Client, Tag, Event, EventBuilder, Filter, HandleNotification, Timestamp, nip04_decrypt, EventId, init_logger, LogLevel
import time
from nostr_sdk.nostr_sdk import Duration, PublicKey

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


class ResultConfig:
    SUPPORTED_TASKS = ["speech-to-text", "translation", "text-to-image", "image-to-image", "image-upscale", "chat"]
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
relay_list = ["wss://relay.damus.io", "wss://blastr.f7z.xyz", "wss://nostr.mutinywallet.com", "wss://relayable.org", "wss://nostr-pub.wellorder.net"]


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

    dmzapfilter = Filter().pubkey(pk).kind(4).kind(9735).since(Timestamp.now())
    dvmfilter = Filter().kind(68001) \
        .kind(65002) \
        .kind(65003) \
        .kind(65004) \
        .kind(65005) \
        .since(Timestamp.now())
    client.subscribe([dmzapfilter, dvmfilter])

    class NotificationHandler(HandleNotification):

        def handle(self, relay_url, event):
            print(f"Received new event from {relay_url}: {event.as_json()}")
            if (65002 <= event.kind() <= 66000) or event.kind() == 68001:  # legacy:
                if isBlackListed(event.pubkey):
                    sendJobStatusReaction(event, "user-blocked-from-service")
                    print("Request by blacklisted user, skipped")
                elif checkTaskisSupported(event):
                    task = getTask(event)
                    if isWhiteListed(event.pubkey().to_hex(), task):
                        print("[Nostr] Whitelisted for task " + task + ". Starting processing..")
                        doWork(event, True)
                    # otherwise send payment request
                    else:
                        bid = 0
                        for tag in event.tags():
                            if tag.as_vec()[0] == 'bid':
                                bid = int(tag.as_vec()[1])

                        amount = 1000000
                        if task == "translation":
                            duration = 1  # todo get task duration
                            amount = ResultConfig.COSTPERUNIT_TRANSLATION * duration * 1000  # *1000 because millisats
                            print("[Nostr][Payment required] New Nostr translation Job event: " + event.as_json())
                        elif task == "speech-to-text":
                            duration = 1  # todo get task duration
                            amount = ResultConfig.COSTPERUNIT_TRANSLATION * duration * 1000  # *1000 because millisats
                            print("[Nostr][Payment required] New Nostr speech-to-text Job event: " + event.as_json())
                        elif task == "text-to-image":
                            amount = ResultConfig.COSTPERUNIT_IMAGEPROCESSING * 1000  # *1000 because millisats
                            print("[Nostr][Payment required] New Nostr generate Image Job event: " + event.as_json())
                        elif task == "image-to-image":
                            # todo get image size
                            amount = ResultConfig.COSTPERUNIT_IMAGEPROCESSING * 1000  # *1000 because millisats
                            print("[Nostr][Payment required] New Nostr convert Image Job event: " + event.as_json())
                        elif task == "image-upscale":
                            amount = ResultConfig.COSTPERUNIT_IMAGEUPSCALING * 1000  # *1000 because millisats
                            print("[Nostr][Payment required] New Nostr upscale Image Job event: " + event.as_json())
                        elif task == "chat":
                            amount = ResultConfig.COSTPERUNIT_IMAGEUPSCALING * 1000  # *1000 because millisats
                            print("[Nostr][Payment required] New Nostr Chat Job event: " + event.as_json())
                        else:
                            print("[Nostr] Task " + task + " is currently not supported by this instance")
                        if bid > 0:
                            willingtopay = bid
                            if willingtopay > ResultConfig.AUTOPROCESS_MIN_AMOUNT * 1000 or willingtopay < ResultConfig.AUTOPROCESS_MAX_AMOUNT * 1000:
                                print("[Nostr][Auto-processing: Payment suspended to end] Job event: " + str(
                                    event.as_json()))
                                doWork(event, False, willingtopay)
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
                    if str(dec_text).startswith("-text-to-image") or str(dec_text).startswith("-speech-to-text") or str(
                            dec_text).startswith("-image-upscale"):
                        time.sleep(2.0)
                        event = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(), "Payment required, please zap this note with at least " + str(ResultConfig.COSTPERUNIT_IMAGEPROCESSING) + " Sats ..", event.id()).to_event(keys)

                        JobstoWatch.append(JobToWatch(id=event.id().to_hex(), timestamp=event.created_at().as_secs(), amount=50,
                                       isPaid=False, status="payment-required", result="", isProcessed=False))
                        client.send_event(event)
                except Exception as e:
                    print(f"Error during content decryption: {e}")
            elif event.kind() == 9735:
                print("Zap received")
                try:
                    for tag in event.tags():
                        if tag.as_vec()[0] == 'bolt11':
                            invoicesats = ParseBolt11Invoice(tag.as_vec()[1])
                        elif tag.as_vec()[0] == 'e':
                            print(tag.as_vec()[1])
                            #zapableevent = getEvent(tag.as_vec()[1])
                            orginalidfilter = Filter().id(tag.as_vec()[1])
                            events = client.get_events_of([orginalidfilter], timedelta(seconds=2))
                            zapableevent = events[0]
                            if (zapableevent.kind() == 65000):  # if a reaction by us got zapped
                                for tag in zapableevent.tags():
                                    amount = 0
                                    if tag.as_vec()[0] == 'amount':
                                        amount = int(tag.as_vec()[0])
                                    elif tag.as_vec()[0] == 'e':
                                        jobidfilter = Filter().id(tag.as_vec()[1])
                                        events = client.get_events_of([jobidfilter], timedelta(seconds=2))
                                        jobevent = events[0]
                                        #jobevent = getEvent(tag.as_vec()[1])
                                        print("[Nostr] Original Job Request event found...")

                                        if (int(amount) <= invoicesats * 1000):
                                            print("[Nostr] Payment-request fulfilled...")
                                            sendJobStatusReaction(jobevent, "payment-accepted")
                                            print(jobevent.id().to_hex())
                                            indices = [i for i, x in enumerate(JobstoWatch) if
                                                       x.id == jobevent.id().to_hex()]
                                            index = -1
                                            if len(indices) > 0:
                                                index = indices[0]
                                            if (index > -1):
                                                # todo also remove ids after x time of waiting, need to store pairs of id / timestamp for that
                                                if (JobstoWatch[index]).isProcessed:  # If payment-required appears after processing
                                                    JobstoWatch[index].isPaid = True
                                                    CheckEventStatus(JobstoWatch[index].result, str(jobevent.as_json()))
                                                elif not (JobstoWatch[
                                                    index]).isProcessed:  # If payment-required appears before processing
                                                    JobstoWatch.pop(index)
                                                    doWork(jobevent, True)
                                        else:
                                            sendJobStatusReaction(jobevent, "payment-rejected", invoicesats * 1000)
                                            print("[Nostr] Invoice was not paid sufficiently")

                            elif zapableevent.kind() == 4:
                                if invoicesats >= ResultConfig.COSTPERUNIT_IMAGEPROCESSING:
                                    print("[Nostr] Original Prompt Job Request event found...")
                                    for tag in zapableevent.tags():
                                        if tag.as_vec()[0] == 'e':
                                            jobidfilter = Filter().id(tag.as_vec()[1])
                                            events = client.get_events_of([jobidfilter], timedelta(seconds=2))
                                            evt = events[0]

                                            indices = [i for i, x in enumerate(JobstoWatch) if x.id == zapableevent.id().to_hex()]
                                            if len(indices) == 1:
                                                event = EventBuilder.new_encrypted_direct_msg(keys, evt.pubkey(), "Payment received, processing started.\n\nI will DM you once your task is ready.", None).to_event(keys)
                                                sendEvent(event)
                                                dec_text = nip04_decrypt(sk, evt.pubkey(), evt.content())
                                                JobstoWatch.pop(indices[0])

                                                if str(dec_text).startswith("-text-to-image"):
                                                    negative_prompt = ""
                                                    prompttemp = dec_text.replace("-text-to-image ", "")
                                                    split = prompttemp.split("-")
                                                    prompt = split[0]
                                                    width = "512"
                                                    height = "512"
                                                    if len(split) > 1:
                                                        for i in split:
                                                            if i.startswith("negative"):
                                                                negative_prompt = i.replace("negative ", "")
                                                            elif i.startswith("width"):
                                                                width = i.replace("width ", "")
                                                            elif i.startswith("height"):
                                                                height = i.replace("height ", "")
                                                        jTag = Tag.parse(["j", "text-to-image"])
                                                        iTag = Tag.parse(["i", prompt, "text"])
                                                        paramTag = Tag.parse(["params", "negative_prompt", negative_prompt])
                                                        paramTag2 = Tag.parse(["params", "size", width, height])
                                                        pTag = Tag.parse(["p", evt.pubkey().to_hex()])
                                                        tags = [jTag, iTag, pTag, paramTag, paramTag2]
                                                        event = EventBuilder(4,"",tags).to_event(keys)
                                                        doWork(event, isPaid=True, isFromBot=True)

                                                elif str(dec_text).startswith("-image-upscale"):
                                                    prompttemp = dec_text.replace("-image-upscale", "")
                                                    split = prompttemp.split("-")
                                                    url = split[0]
                                                    if len(split) > 1:
                                                        for i in split:
                                                            if i.startswith("prompt"):
                                                                prompt = i.replace("url ", "")
                                                            elif i.startswith("negative"):
                                                                negative_prompt = i.replace("negative ", "")
                                                        jTag = Tag.parse(["j", "image-upscale"])
                                                        iTag = Tag.parse(["i", url, "url"])
                                                        pTag = Tag.parse(["p", evt.pubkey().to_hex()])
                                                        tags = [jTag, iTag, pTag]
                                                        event = EventBuilder(4, "", tags).to_event(keys)
                                                        doWork(event, isPaid=True, isFromBot=True)

                                                elif str(dec_text).startswith("-speech-to-text"):
                                                     url = dec_text.replace("-speech-to-text ", "")
                                                     jTag = Tag.parse(["j", "speech-to-text"])
                                                     iTag = Tag.parse(["i", url, "url"])
                                                     oTag = Tag.parse(["output", "text/plain"])
                                                     pTag = Tag.parse(["p", evt.pubkey().to_hex()])
                                                     tags = [jTag, iTag, oTag, pTag]
                                                     event = EventBuilder(4, "", tags).to_event(keys)
                                                     doWork(event, isPaid=True, isFromBot=True)

                                                elif str(dec_text).startswith("-chat"):
                                                    text = dec_text.replace("-chat ", "")
                                                    jTag = Tag.parse(["j", "chat"])
                                                    iTag = Tag.parse(["i", text, "text"])
                                                    oTag = Tag.parse(["output", "text/plain"])
                                                    pTag = Tag.parse(["p", evt.pubkey().to_hex()])
                                                    tags = [jTag, iTag, oTag, pTag]
                                                    event = EventBuilder(4, "", tags).to_event(keys)
                                                    doWork(event, isPaid=True, isFromBot=True)

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
        if (isBot):
            request_form["isBot"] = "True"
        else:
            request_form["isBot"] = "False"

        # defaults might be overwritten by nostr event
        alignment = "raw"
        request_form["startTime"] = "0"
        request_form["endTime"] = "0"

        for tag in event.tags():
            if tag.as_vec()[0] == 'params':
                print(tag.as_vec())
                param = tag.as_vec()[1]
                if param == "range":  # check for paramtype
                    request_form["startTime"] = re.sub('\D', '', tag.as_vec()[2])
                    request_form["endTime"] = re.sub('\D', '', tag.as_vec()[3])
                elif param == "alignment":  # check for paramtype
                    alignment = tag.as_vec()[2]
                elif param == "length":  # check for paramtype
                    length = tag.as_vec()[2]
                elif param == "language":  # check for paramtype
                    translation_lang = str(tag.as_vec()[2]).split('-')[0]

        if task == "speech-to-text":
            # Declare specific model type e.g. whisperx_large-v2
            request_form["mode"] = "PREDICT"
            modelopt = "large-v2"

            request_form["schemeType"] = "FREE"
            request_form["scheme"] = "transcript"
            request_form["streamName"] = "audio"
            request_form["trainerFilePath"] = 'models\\trainer\\' + str(
                request_form["schemeType"]).lower() + '\\' + str(
                request_form["scheme"]) + '\\audio{audio}\\whisperx\\whisperx_transcript.trainer'
            request_form["optStr"] = 'model=' + modelopt + ';alignment_mode=' + alignment + ';batch_size=2'

        elif task == "translation":
            request_form["mode"] = "PREDICT_STATIC"
            # outsource this to its own script, ideally. This is not using the database for now, but probably should.
            inputtype = "event"
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    inputtype = tag.as_vec()[2]
                    break
            if inputtype == "event":
                for tag in event.tags():
                    if tag.as_vec()[0] == 'i':
                        jobidfilter = Filter().id(tag.as_vec()[1])
                        events = client.get_events_of([jobidfilter], timedelta(seconds=5))
                        evt = events[0]
                        #evt = getEvent(sourceid)
                        text = evt.content()
                        break

            elif inputtype == "text":
                for tag in event.tags():
                    if tag.as_vec()[0] == 'i':
                        text = tag.as_vec()[1]
                        break
            request_form["optStr"] = 'text=' + text + ';translation_lang=' + translation_lang
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
        elif task == "image-to-image":
            request_form["mode"] = "PREDICT_STATIC"
            prompt = ""
            negative_prompt = ""
            strength = 0.75
            guidance_scale = 7.5

            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    type = tag.as_vec()[2]
                    if type == "url":
                        url = tag.as_vec()[1]
                elif tag.as_vec()[0] == 'params':
                    if tag.as_vec()[1] == "prompt":  # check for paramtype
                        prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "negative_prompt":  # check for paramtype
                        negative_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "strength":  # check for paramtype
                        strength = float(tag.as_vec()[2])
                    elif tag.as_vec()[1] == "guidance_scale":  # check for paramtype
                        guidance_scale = float(tag.as_vec()[2])
            request_form[
                "optStr"] = 'url=' + url + ';prompt=' + prompt + ';negative_prompt=' + negative_prompt + ';strength=' + strength + ';guidance_scale=' + guidance_scale

        elif task == "text-to-image":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'text-to-image'
            width = "512"
            height = "512"
            extra_prompt = ""
            negative_prompt = ""

            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    type = tag.as_vec()[2]
                    if type == "text":
                        prompt = tag.as_vec()[1]
                    elif type == "event":
                        jobidfilter = Filter().id(tag.as_vec()[1])
                        events = client.get_events_of([jobidfilter], timedelta(seconds=5))
                        evt = events[0]
                        # evt = getEvent(sourceid)
                        prompt = evt.content()
                elif tag.as_vec()[0] == 'params':
                    if tag.as_vec()[1] == "extra_prompt":  # check for paramtype
                        extra_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "negative_prompt":  # check for paramtype
                        negative_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "size":  # check for paramtype
                        width = tag.as_vec()[2]
                        height = tag.as_vec()[3]
            request_form["optStr"] = 'prompt=' + prompt + ';extra_prompt=' + extra_prompt + ';negative_prompt=' + negative_prompt + ';width=' + width + ';height=' + height

        elif task == "image-upscale":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'image-upscale'
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    type = tag.as_vec()[2]
                    if type == "url":
                        url = tag.as_vec()[1]
                        request_form["optStr"] = 'url=' + url

        return request_form

    def doWork(Jobevent, isPaid, amount=0, isFromBot=False):
        if (Jobevent.kind() >= 65002 and Jobevent.kind() <= 66000) or Jobevent.kind() == 68001 or Jobevent.kind() == 4:
            request_form = createRequestFormfromNostrEvent(Jobevent, isFromBot)
            task = getTask(Jobevent)
            print(task)

            if task == "speech-to-text":
                print("[Nostr] Adding Nostr speech-to-text Job event: " + Jobevent.as_json())
                organizeInputData(Jobevent, request_form)
            elif task == "event-list-generation" or task == "summarization" or task == "chat" or task.startswith(
                    "unknown"):
                print("Task not (yet) supported")
                return
            else:
                print("[Nostr] Adding " + task + " Job event: " + Jobevent.as_json())

            if (Jobevent.kind() >= 65002 and Jobevent.kind() < 66000) or Jobevent.kind() == 68001:
                sendJobStatusReaction(Jobevent, "started", isPaid, amount)
            url = 'http://' + os.environ["NOVA_HOST"] + ':' + os.environ["NOVA_PORT"] + '/' + str(
                request_form["mode"]).lower()
            headers = {'Content-type': 'application/x-www-form-urlencoded'}
            requests.post(url, headers=headers, data=request_form)
    client.handle_notifications(NotificationHandler())

    while True:
        time.sleep(5.0)


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
                if (o.query.find('t') != -1):
                    request_form["startTime"] = q['t'][0]  # overwrite from link.. why not..
                    print("Setting start time automatically to " + request_form["startTime"])
                    if float(request_form["endTime"]) > 0.0:
                        request_form["endTime"] = str(float(q['t'][0]) + float(request_form["endTime"]))
                        print("Moving end time automatically to " + request_form["endTime"])
            except Exception:
                print("video not available")
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
                sendJobStatusReaction(event, "format not supported")
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
            sendJobStatusReaction(event, "failed")
            return

        print("Duration of the Media file: " + str(duration))
        if float(request_form['endTime']) == 0.0:
            end_time = duration
        elif float(request_form['endTime']) > duration:
            end_time = duration
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
    pablof7z = "fa984bd7dbb282f07e16e7ae87b26a2a7b9b90b7246a44771f0cf5ae58018f52"
    localnostrtest = "558497db304332004e59387bc3ba1df5738eac395b0e56b45bfb2eb5400a1e39"
    # PublicKey.from_npub("npub...").hex()
    whitelsited_npubs_speechtotext = []  # remove this to test LN Zaps
    whitelsited_npubs_translation = [localnostrtest]
    whitelsited_npubs_texttoimage = [localnostrtest]
    whitelsited_npubs_imagetoimage = [localnostrtest]
    whitelsited_npubs_imageupscale = [localnostrtest]
    whitelsited_npubs_chat = [localnostrtest]

    whitelsited_all_tasks = []

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
            return
    elif (task == "image-upscale"):
        if any(pubkey == c for c in whitelsited_npubs_imageupscale) or any(
                pubkey == c for c in whitelsited_all_tasks):
            return True
    elif (task == "chat"):
        if any(pubkey == c for c in whitelsited_npubs_chat) or any(pubkey == c for c in whitelsited_all_tasks):
            return True
    return False
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
def getTask(event):
        if event.kind() == 68001:  # legacy
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
            return "speech-to-text"
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
            if ResultConfig.SHOWRESULTBEFOREPAYMENT and not isPaid:
                sendNostrReplyEvent(content, originaleventstr)
                sendJobStatusReaction(originalevent, "success", amount)  # or payment-required, or both?
            elif not ResultConfig.SHOWRESULTBEFOREPAYMENT and not isPaid:
                sendJobStatusReaction(originalevent, "success", amount)  # or payment-required, or both?

            if (ResultConfig.SHOWRESULTBEFOREPAYMENT and isPaid):
                JobstoWatch.remove(x)
            elif not ResultConfig.SHOWRESULTBEFOREPAYMENT and isPaid:
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
            event = EventBuilder.new_encrypted_direct_msg(keys, reckey,
                                                          "Your Result: \n\n" + resultcontent, None).to_event(keys)
            sendEvent(event)

        else:
            sendNostrReplyEvent(resultcontent, originaleventstr)
            sendJobStatusReaction(originalevent, "success")
def sendNostrReplyEvent(content, originaleventstr):
    # Once the Job is finished we reply with the results with a 68002 event
    originalevent = Event.from_json(originaleventstr)


    requesttag = Tag.parse(["request", originaleventstr.replace("\\", "")])
    etag = Tag.parse(["e", originalevent.id().to_hex()])
    ptag = Tag.parse(["p", originalevent.pubkey().to_hex()])
    statustag = Tag.parse(["status", "success"])

    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    event = EventBuilder(65001, str(content), [requesttag, etag, ptag, statustag]).to_event(keys)
    sendEvent(event)

    print("[Nostr] 65001 Job Response event sent: " + event.as_json())

    return event.as_json()
def sendJobStatusReaction(originalevent, status, isPaid=True, amount=0):
        if status == "started":
            reaction = emoji.emojize(":thumbs_up:")
        elif status == "success":
            reaction = emoji.emojize(":call_me_hand:")
        elif status == "failed":
            reaction = emoji.emojize(":thumbs_down:")
        elif status == "payment-required":
            reaction = emoji.emojize(":orange_heart:")
        elif status == "payment-accepted":
            reaction = emoji.emojize(":smiling_face_with_open_hands:")
        elif status == "payment-rejected":
            reaction = emoji.emojize(":see_no_evil_monkey:")
        elif status == "user-blocked-from-service":
            reaction = emoji.emojize(":see_no_evil_monkey:")
        else:
            reaction = emoji.emojize(":see_no_evil_monkey:")

        etag = Tag.parse(["e", originalevent.id().to_hex()])
        ptag = Tag.parse(["p", originalevent.pubkey().to_hex()])
        statustag = Tag.parse(["status", status])
        tags = [etag, ptag, statustag]

        if status == "success" or status == "failed":  #
            for x in JobstoWatch:
                if x.id == originalevent.id():
                    isPaid = x.isPaid
                    amount = x.amount
                    break
        if status == "payment-required" or (status == "started" and not isPaid):
            JobstoWatch.append(
                JobToWatch(id=originalevent.id().to_hex(), timestamp=originalevent.created_at().as_secs(), amount=amount, isPaid=isPaid,
                           status=status, result="", isProcessed=False))
            print(str(JobstoWatch))
        if status == "payment-required" or (status == "started" and not isPaid) or (status == "success" and not isPaid):
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
                for name in content["name"]:
                    clearedName = str(name).lstrip("\'").rstrip("\'")
                    result = result + clearedName + "\n"
                content = str(result).replace("\"", "").replace('[', "").replace(']', "").lstrip(None)
            # TODO add more

    return content
def checkTaskisSupported(event):
    task = getTask(event)
    print("Received new Task: " + task)
    hasitag = False
    for tag in event.tags():
        if tag.as_vec()[0] == 'i':
            input = tag.as_vec()[1]
            inputtype = tag.as_vec()[2]
            hasitag = True

        elif tag.as_vec()[0] == 'output':
            output = tag.as_vec()[1]
            if output != "text/plain":
                return False

    if not hasitag:
        return False

    if task not in ResultConfig.SUPPORTED_TASKS:  # The Tasks this DVM supports (can be extended)
        return False
    if task == "translation" and (
            inputtype != "event" and inputtype != "job" and inputtype != "text"):  # The input types per task
        return False
    if task == "translation" and len(event.content) > 4999:  # Google Services have a limit of 5000 signs
        return False
    if task == "speech-to-text" and inputtype != "url":  # The input types per task
        return False
    if task == "image-upscale" and inputtype != "url":  # The input types per task
        return False
    if inputtype == 'url' and not CheckUrlisReadable(input):
        return False

    return True
def CheckUrlisReadable(url):
    if not str(url).startswith("http"):
        return False
    # If it's a YouTube oder Overcast link, we suppose we support it
    if str(url).replace("http://", "").replace("https://", "").replace("www.", "").replace("youtu.be/",
                                                                                           "youtube.com?v=")[
       0:11] == "youtube.com" and str(url).find("live") == -1:
        return (checkYoutubeLinkValid(url))  # not live, protected etc
    elif str(url).startswith("https://overcast.fm/"):
        return True

    # If link is comaptible with one of these file formats, it's fine.
    req = requests.get(url)
    content_type = req.headers['content-type']
    if content_type == 'audio/x-wav' or str(url).endswith(".wav") \
            or content_type == 'audio/mpeg' or str(url).endswith(".mp3") \
            or content_type == 'image/png' or str(url).endswith(".png") \
            or content_type == 'image/jpg' or str(url).endswith(".jpg") \
            or content_type == 'image/jpeg' or str(url).endswith(".jpeg") \
            or content_type == 'audio/ogg' or str(url).endswith(".ogg") \
            or content_type == 'video/mp4' or str(url).endswith(".mp4") \
            or content_type == 'video/avi' or str(url).endswith(".avi") \
            or content_type == 'video/mov' or str(url).endswith(".mov"):
        return True

    # Otherwise we will not offer to do the job.
    return False
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

if __name__ == '__main__':
    os.environ["NOVA_DATA_DIR"] = "W:\\nova\\data"
    os.environ["NOVA_NOSTR_KEY"] = "privkey"
    os.environ["NOVA_HOST"] = "127.0.ß.1"
    os.environ["NOVA_PORT"] = "27017"
    nostr_client()


