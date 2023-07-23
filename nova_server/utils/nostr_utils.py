import base64
import os
import urllib
from urllib.parse import urlparse

from decord import AudioReader, cpu
from pynostr.base_relay import RelayPolicy
from pynostr.encrypted_dm import EncryptedDirectMessage

from pynostr.relay_list import RelayList

import ffmpegio
from pynostr.relay_manager import RelayManager
from pynostr.filters import FiltersList, Filters
from pynostr.event import EventKind
import json
from pynostr.event import Event
from pynostr.key import PrivateKey, PublicKey
from dataclasses import dataclass
import emoji
import re

from sympy.ntheory.primetest import is_lucas_prp

from nova_server.utils.db_utils import add_new_session_to_db, db_entry_exists
from nova_server.utils.mediasource_utils import download_podcast, downloadYouTube, checkYoutubeLinkValid
from configparser import ConfigParser

import time
import datetime
import requests

import uuid

# TODO
# check expiry of tasks/available output format/model/ (task is checked already). if not available ignore the job,
# send reaction on error (send sats back ideally, library meh, same for under payment),
# send reaction processing-scheduled when task is waiting for previous task to finish, max limit to wait?
# store whitelist (and maybe a blacklist) in a config/db
# clear list of  tasks (JobstoWatch) to watch after some time (timeout if invoice not paid),
# consider max-sat amount at all,
# consider reactions from customers (Kind 68003 event)
# consider encrypted DMS with tasks (decrypt seems broken in pynostr, or dependency version of)
# add more output formats (webvtt, srt)
# refactor translate to own module
# add summarization task (GPT4all?, OpenAI api?) in own module

# purge database and files from time to time?

sinceLastNostrUpdate = int(datetime.datetime.now().timestamp())


class ResultConfig:
    SUPPORTED_TASKS = ["speech-to-text", "translation", "text-to-image", "image-to-image", "image-upscale"]
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
lastdm = ""
lastzap = ""
lastdvm = ""

rl = RelayList()
url_list = ["wss://relay.damus.io", "wss://relay.snort.social",
            "wss://blastr.f7z.xyz",
            "wss://nostr.mutinywallet.com", "wss://relayable.org"]
ignore_url_list = ["wss://nostr-pub.wellorder.net"]
relaytimeout = 4
policy = RelayPolicy()
rl.append_url_list(url_list, policy)


def nostrReceiveAndManageNewEvents():
    privkey = PrivateKey.from_hex(os.environ["NOVA_NOSTR_KEY"])
    global sinceLastNostrUpdate
    global JobstoWatch

    relay_manager = RelayManager(timeout=relaytimeout)
    relay_manager.add_relay_list(rl)

    # print("[Nostr] Listen to new events since: " + str(sinceLastNostrUpdate))

    vendingFilter = Filters(kinds=[68001], since=sinceLastNostrUpdate, limit=20)
    zapFilter = Filters(kinds=[EventKind.ZAPPER], limit=20, since=sinceLastNostrUpdate)
    zapFilter.add_arbitrary_tag('p', [privkey.public_key.hex()])
    # zapFilter.add_arbitrary_tag('e', IDstoWatch)
    # reactFilter = Filters(kinds=[EventKind.REACTION], limit=20, since=sinceLastNostrUpdate)
    # reactFilter.add_arbitrary_tag('p', privkey.public_key.hex())
    dmFilter = Filters(kinds=[EventKind.ENCRYPTED_DIRECT_MESSAGE], limit=5, since=sinceLastNostrUpdate)
    dmFilter.add_arbitrary_tag('p', [privkey.public_key.hex()])
    filters = FiltersList([vendingFilter, zapFilter, dmFilter])
    subscription_id = uuid.uuid1().hex
    relay_manager.add_subscription_on_all_relays(subscription_id, filters)
    relay_manager.run_sync()

    while relay_manager.message_pool.has_events():
        event_msg = relay_manager.message_pool.get_event()

        sinceLastNostrUpdate = int(max(event_msg.event.created_at + 1, sinceLastNostrUpdate))
        event = event_msg.event
        # check for Task, for this demo use case only get active when task is speech-to-text
        if event.kind == 68001:
            global lastdvm
            if (event.id != lastdvm):
                lastdvm = event.id
                # if npub sending the 68001 event is whitelisted, we just do the work
                if isBlackListed(event.pubkey):
                    sendJobStatusReaction(event, "user-blocked-from-service")
                    print("Request by blacklisted user, skipped")

                elif checkTaskisSupported(event):

                    if isWhiteListed(event.pubkey, event.get_tag_list('j')[0][0]):
                        print(
                            "[Nostr] Whitelisted for task " + event.get_tag_list('j')[0][0] + ". Starting processing..")
                        doWork(event, True)
                    # otherwise send payment request
                    else:
                        amount = 1000000
                        if event.get_tag_list('j')[0][0] == "translation":
                            duration = 1  # todo get task duration
                            amount = ResultConfig.COSTPERUNIT_TRANSLATION * duration * 1000  # *1000 because millisats
                            print("[Nostr][Payment required] New Nostr translation Job event: " + str(event.to_dict()))
                        elif event.get_tag_list('j')[0][0] == "speech-to-text":
                            duration = 1  # todo get task duration
                            amount = ResultConfig.COSTPERUNIT_TRANSLATION * duration * 1000  # *1000 because millisats
                            print(
                                "[Nostr][Payment required] New Nostr speech-to-text Job event: " + str(event.to_dict()))
                        elif event.get_tag_list('j')[0][0] == "text-to-image":
                            amount = ResultConfig.COSTPERUNIT_IMAGEPROCESSING * 1000  # *1000 because millisats
                            print(
                                "[Nostr][Payment required] New Nostr generate Image Job event: " + str(event.to_dict()))
                        elif event.get_tag_list('j')[0][0] == "image-to-image":
                            amount = ResultConfig.COSTPERUNIT_IMAGEPROCESSING * 1000  # *1000 because millisats
                            print(
                                "[Nostr][Payment required] New Nostr convert Image Job event: " + str(event.to_dict()))
                        elif event.get_tag_list('j')[0][0] == "image-upscale":
                            amount = ResultConfig.COSTPERUNIT_IMAGEUPSCALING * 1000  # *1000 because millisats
                            print(
                                "[Nostr][Payment required] New Nostr upscale Image Job event: " + str(event.to_dict()))
                        else:
                            print("[Nostr] Task " + event.get_tag_list('j')[0][
                                0] + " is currently not supported by this instance")

                        if len(event.get_tag_list("bid")) > 0:
                            willingtopay = int(event.get_tag_list("bid")[0][0])
                            if willingtopay > ResultConfig.AUTOPROCESS_MIN_AMOUNT * 1000 or willingtopay < ResultConfig.AUTOPROCESS_MAX_AMOUNT * 1000:
                                print("[Nostr][Auto-processing: Payment suspended to end] Job event: " + str(
                                    event.to_dict()))
                                doWork(event, False, willingtopay)
                            else:
                                if willingtopay >= amount:
                                    sendJobStatusReaction(event, "payment-required", False,
                                                          willingtopay)  # Take what user is willing to pay, min server rate
                                else:
                                    sendJobStatusReaction(event, "payment-rejected", False,
                                                          amount)  # Reject and tell user minimum amount

                        else:  # If there is no bid, just request server rate from user
                            print("[Nostr] Requesting payment for Event: " + event.id)
                            sendJobStatusReaction(event, "payment-required", False, amount)
                else:
                    print("Got new Task but can't process it, skipping..")

        elif event.kind == EventKind.ZAPPER:  # 9735
            global lastzap
            if (event.id != lastzap):
                lastzap = event.id
                print("[Nostr]Zap Received")
                # Zaps to us
                lninvoice = event.get_tag_list('bolt11')[0][0]
                invoicesats = ParseBolt11Invoice(lninvoice)
                print("[Nostr]Zap Received: " + str(event.to_dict()))
                eventid = event.get_tag_list("e")[0][0]

                # Get specific reaction event
                relay_manager2 = RelayManager(timeout=relaytimeout)
                relay_manager2.add_relay_list(rl)
                filters = FiltersList([Filters(ids=[eventid], limit=1)])
                subscription_id = uuid.uuid1().hex
                relay_manager2.add_subscription_on_all_relays(subscription_id, filters)
                relay_manager2.run_sync()
                zapableevent = relay_manager2.message_pool.get_event().event
                relay_manager2.close_all_relay_connections()

                if (zapableevent.kind == 68003):  # if a reaction by us got zapped
                    if (int(zapableevent.get_tag_list('amount')[0][0]) <= invoicesats * 1000):
                        print("[Nostr] Payment-request fulfilled...")
                        event68001id = zapableevent.get_tag_list('e')[0][0]
                        relay_manager3 = RelayManager(timeout=relaytimeout)
                        relay_manager3.add_relay_list(rl)
                        filters = FiltersList([Filters(ids=[event68001id], kinds=[68001], limit=1)])

                        subscription_id = uuid.uuid1().hex
                        relay_manager3.add_subscription_on_all_relays(subscription_id, filters)
                        relay_manager3.run_sync()

                        event68001 = relay_manager3.message_pool.get_event().event
                        print("[Nostr] Original 68001 Job Request event found...")
                        relay_manager3.close_all_relay_connections()

                        sendJobStatusReaction(event68001, "payment-accepted")
                        indices = [i for i, x in enumerate(JobstoWatch) if x.id == event68001.id]
                        index = -1
                        if len(indices) > 0:
                            index = indices[0]
                        if (index > -1):
                            # todo also remove ids after x time of waiting, need to store pairs of id / timestamp for that
                            if (JobstoWatch[index]).isProcessed:  # If payment-required appears after processing
                                JobstoWatch[index].isPaid = True
                                CheckEventStatus(JobstoWatch[index].result, str(event68001.to_dict()))
                            elif not (JobstoWatch[index]).isProcessed:  # If payment-required appears before processing
                                JobstoWatch.pop(index)
                                doWork(event68001, True)
                    else:
                        sendJobStatusReaction(event68001, "payment-rejected", invoicesats * 1000)

                        print("[Nostr] Invoice was not paid sufficiently")
                elif zapableevent.kind == EventKind.ENCRYPTED_DIRECT_MESSAGE:
                    if invoicesats >= ResultConfig.COSTPERUNIT_IMAGEPROCESSING:

                        eventid = event.get_tag_list("e")[0][0]

                        # Get specific reaction event
                        relay_manager2 = RelayManager(timeout=relaytimeout)
                        relay_manager2.add_relay_list(rl)
                        filters = FiltersList([Filters(ids=[eventid], limit=1)])
                        subscription_id = uuid.uuid1().hex
                        relay_manager2.add_subscription_on_all_relays(subscription_id, filters)
                        relay_manager2.run_sync()
                        zapableevent = relay_manager2.message_pool.get_event().event
                        relay_manager2.close_all_relay_connections()
                        relay_manager3 = RelayManager(timeout=relaytimeout)
                        relay_manager3.add_relay_list(rl)
                        filters = FiltersList([Filters(ids=[zapableevent.id], limit=1)])

                        subscription_id = uuid.uuid1().hex
                        relay_manager3.add_subscription_on_all_relays(subscription_id, filters)
                        relay_manager3.run_sync()

                        event68001 = relay_manager3.message_pool.get_event().event
                        prompteventid = event68001.get_tag_list("e")[0][0]

                        relay_manager3.close_all_relay_connections()
                        relay_manager4 = RelayManager(timeout=relaytimeout)
                        relay_manager4.add_relay_list(rl)
                        filters = FiltersList([Filters(ids=[prompteventid], limit=1)])

                        subscription_id = uuid.uuid1().hex
                        relay_manager4.add_subscription_on_all_relays(subscription_id, filters)
                        relay_manager4.run_sync()

                        promptevent = relay_manager4.message_pool.get_event().event
                        print("[Nostr] Original Prompt Job Request event found...")
                        relay_manager4.close_all_relay_connections()

                        indices = [i for i, x in enumerate(JobstoWatch) if x.id == promptevent.id]
                        if len(indices) == 1:
                            sendDM(privkey.hex(), promptevent.pubkey,
                                   "Payment received, processing started.\n\nI will DM you once your image is ready.")
                            dec_text = decryptDM(privkey, promptevent.pubkey, promptevent.content)
                            print(dec_text)
                            if str(dec_text).startswith("-text-to-image"):
                                negative_prompt = ""
                                prompttemp = dec_text.replace("-text-to-image ", "")
                                split = prompttemp.split("-negative")
                                prompt = split[0]
                                if len(split) > 1:
                                    negative_prompt = split[1]

                                promptevent.add_tag('j', ["text-to-image"])
                                promptevent.add_tag('i', [prompt, "text"])
                                promptevent.add_tag('params', ["negative_prompt", negative_prompt])
                                JobstoWatch.pop(indices[0])
                                doWork(promptevent, isPaid=True, isFromBot=True)

                else:
                    print("[Nostr] Zap was not for a kind 68003 or 4 reaction, skipping")

        elif event.kind == 68003:
            print("[Nostr]Reaction Received: " + str(event.to_dict()))

        elif event.kind == EventKind.ENCRYPTED_DIRECT_MESSAGE:
            # BOT MODE
            global lastdm
            if (event.id != lastdm):
                lastdm = event.id
                dec_text = decryptDM(privkey, event.pubkey, event.content)
                print(dec_text)
                if str(dec_text).startswith("-text-to-image"):
                    sendDM(privkey.hex(), event.pubkey, "Payment required, please zap this note with at least " + str(
                        ResultConfig.COSTPERUNIT_IMAGEPROCESSING) + " Sats ..", event.id)
                    JobstoWatch.append(JobToWatch(id=event.id, timestamp=event.created_at, amount=50, isPaid=False,
                                                  status="payment-required", result="", isProcessed=False))

    relay_manager.close_all_relay_connections()


def sendDM(privkey, pubkey, message, replytoid=""):
    dm = EncryptedDirectMessage()
    dm.encrypt(privkey,
               recipient_pubkey=pubkey,
               cleartext_content=message
               )
    dm_event = dm.to_event()
    if (replytoid != ""):
        dm_event.add_event_ref(replytoid)
        # create 'p' tag reference to the pubkey you're replying to
        dm_event.add_pubkey_ref(pubkey)
    dm_event.sign(privkey)
    relay_managers = RelayManager(timeout=relaytimeout)
    relay_managers.add_relay_list(rl)
    relay_managers.publish_event(dm_event)
    relay_managers.run_sync()
    time.sleep(3)
    relay_managers.close_all_relay_connections()


def decryptDM(privkey, pubkey, content):
    from . import cbc
    if "?iv=" in content:
        shared_secret = privkey.compute_shared_secret(pubkey)
        aes = cbc.AESCipher(key=shared_secret)
        enc_text_b64, iv_b64 = content.split("?iv=")
        iv = base64.decodebytes(iv_b64.encode("utf-8"))
        enc_text = base64.decodebytes(enc_text_b64.encode("utf-8"))
        dec_text = aes.decrypt(iv, enc_text)
        return dec_text
    return "Couldn't decrypt the message"


def doWork(event68001, isPaid, amount=0, isFromBot = False):
    if event68001.kind == 68001 or event68001.kind == 4:
        request_form = createRequestFormfromNostrEvent(event68001, isFromBot)
        if event68001.get_tag_list('j')[0][0] == "speech-to-text":
            print("[Nostr] Adding Nostr speech-to-text Job event: " + str(event68001.to_dict()))
            organizeInputData(event68001, request_form)
        elif event68001.get_tag_list('j')[0][0] == "translation":
            print("[Nostr] Adding translation Job event: " + str(event68001.to_dict()))
        elif event68001.get_tag_list('j')[0][0] == "text-to-image":
            print("[Nostr] Adding Image Generation Job event: " + str(event68001.to_dict()))
        elif event68001.get_tag_list('j')[0][0] == "image-to-image":
            print("[Nostr] Adding Image Conversion Job event: " + str(event68001.to_dict()))
        elif event68001.get_tag_list('j')[0][0] == "image-upscale":
            print("[Nostr] Adding Image Upscale Job event: " + str(event68001.to_dict()))
        else:
            print("[Nostr] Task " + event68001.get_tag_list('j')[0][0] + " is currently not supported by this instance")
        if event68001.kind == 68001:
            sendJobStatusReaction(event68001, "started", isPaid, amount)

        url = 'http://' + os.environ["NOVA_HOST"] + ':' + os.environ["NOVA_PORT"] + '/' + str(
            request_form["mode"]).lower()
        headers = {'Content-type': 'application/x-www-form-urlencoded'}
        requests.post(url, headers=headers, data=request_form)

def checkTaskisSupported(event):
    if (len(event.get_tag_list('j')) < 1 or len(event.get_tag_list('i')) < 1):
        return False
    task = event.get_tag_list('j')[0][0]
    url = event.get_tag_list('i')[0][0]
    inputtype = event.get_tag_list('i')[0][1]
    content = event.content

    if task not in ResultConfig.SUPPORTED_TASKS:  # The Tasks this DVM supports (can be extended)
        return False
    if task == "translation" and (inputtype != "event" and inputtype != "job"):  # The input types per task
        return False
    if task == "translation" and len(content) > 4999:  # Google Services have a limit of 5000 signs
        return False
    if task == "speech-to-text" and inputtype != "url":  # The input types per task
        return False
    if task == "image-upscale" and inputtype != "url":  # The input types per task
        return False
    if inputtype == 'url' and not CheckUrlisReadable(url):
        return False
    if len(event.get_tag_list(
            'output')) > 0:  # if output tag is set check for available formats, else use server default output
        if event.get_tag_list('output')[0][0] != "text/plain":  # or..
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


def organizeInputData(event, request_form):
    data_dir = os.environ["NOVA_DATA_DIR"]
    session = event.id

    inputtype = "url"
    if len(event.get_tag_list('i')[0]) > 1:
        inputtype = event.get_tag_list('i')[0][1]

    if inputtype == "url":
        url = event.get_tag_list('i')[0][0]
        if not os.path.exists(data_dir + '\\' + request_form["database"] + '\\' + session):
            os.mkdir(data_dir + '\\' + request_form["database"] + '\\' + session)
        # We can support some services that don't use default media links, like overcastfm for podcasts
        if str(url).startswith("https://overcast.fm/"):
            filename = data_dir + '\\' + request_form["database"] + '\\' + session + '\\' + request_form[
                "roles"] + ".originalaudio.mp3"
            print("Found overcast.fm Link.. downloading")
            download_podcast(url, filename)
            finaltag = str(url).replace("https://overcast.fm/", "").split('/')

            if (len(finaltag) > 1):
                t = time.strptime(finaltag[1], "%H:%M:%S")
                seconds = t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec
                request_form["startTime"] = str(seconds)  # overwrite from link.. why not..
                print("Setting start time automatically to " + request_form["startTime"])
                if float(request_form["endTime"]) > 0.0:
                    request_form["endTime"] = seconds + float(request_form["endTime"])
                    print("Moving end time automatically to " + request_form["endTime"])
        # is youtube link?
        elif str(url).replace("http://", "").replace("https://", "").replace("www.", "").replace("youtu.be/",
                                                                                                 "youtube.com?v=")[
             0:11] == "youtube.com":
            filepath = data_dir + '\\' + request_form["database"] + '\\' + session + '\\'
            try:
                filename = downloadYouTube(url, filepath)
                o = urlparse(url)
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
            req = requests.get(url)
            content_type = req.headers['content-type']
            if content_type == 'audio/x-wav' or str(url).endswith(".wav"):
                ext = "wav"
                type = "audio"
            elif content_type == 'audio/mpeg' or str(url).endswith(".mp3"):
                ext = "mp3"
                type = "audio"
            elif content_type == 'audio/ogg' or str(url).endswith(".ogg"):
                ext = "ogg"
                type = "audio"
            elif content_type == 'video/mp4' or str(url).endswith(".mp4"):
                ext = "mp4"
                type = "video"
            elif content_type == 'video/avi' or str(url).endswith(".avi"):
                ext = "avi"
                type = "video"
            elif content_type == 'video/mov' or str(url).endswith(".mov"):
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


def createRequestFormfromNostrEvent(event, isBot=False):
    # Only call this if config is not available, adjust function to your db
    # savConfig()
    if len(event.get_tag_list('j')) > 0:
        task = event.get_tag_list('j')[0][0]

    # Read config.ini file
    config_object = ConfigParser()
    config_object.read("nostrconfig.ini")
    if len(config_object) == 1:
        dbUser = input("Please enter a DB User:\n")
        dbPassword = input("Please enter DB User Password:\n")
        dbServer = input("Please enter a DB Host:\n")
        savConfig(dbUser, dbPassword, dbServer, "nostr_test", "nostr", "system")
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
    request_form["jobID"] = event.id

    request_form["frameSize"] = 0
    request_form["stride"] = request_form["frameSize"]
    request_form["leftContext"] = 0
    request_form["rightContext"] = 0
    request_form["nostrEvent"] = str(event.to_dict())
    request_form["sessions"] = event.id
    request_form["isBot"] = isBot

    # defaults might be overwritten by nostr event
    alignment = "raw"
    request_form["startTime"] = 0
    request_form["endTime"] = 0

    params = event.get_tag_list('params')
    for param in params:
        if param[0] == "range":  # check for paramtype
            request_form["startTime"] = re.sub('\D', '', param[1])
            request_form["endTime"] = re.sub('\D', '', param[2])
        elif param[0] == "alignment":  # check for paramtype
            alignment = param[1]
        elif param[0] == "length":  # check for paramtype
            length = param[1]
        elif param[0] == "language":  # check for paramtype
            translation_lang = str(param[1]).split('-')[0]

    if task == "speech-to-text":
        # Declare specific model type e.g. whisperx_large-v2
        request_form["mode"] = "PREDICT"
        if len(event.get_tag_list('j')[0]) > 1:
            model = event.get_tag_list('j')[0][1]
            modelopt = str(model).split('_')[1]
        else:
            modelopt = "large-v2"

        request_form["schemeType"] = "FREE"
        request_form["scheme"] = "transcript"
        request_form["streamName"] = "audio"
        request_form["trainerFilePath"] = 'models\\trainer\\' + str(request_form["schemeType"]).lower() + '\\' + str(
            request_form["scheme"]) + '\\audio{audio}\\whisperx\\whisperx_transcript.trainer'
        request_form["optStr"] = 'model=' + modelopt + ';alignment_mode=' + alignment + ';batch_size=2'



    elif task == "translation":
        request_form["mode"] = "PREDICT_STATIC"
        # outsource this to its own script, ideally. This is not using the database for now, but probably should.
        inputtype = "event"
        if len(event.get_tag_list('i')[0]) > 1:
            inputtype = event.get_tag_list('i')[0][1]
        if inputtype == "event":
            sourceid = event.get_tag_list('i')[0][0]
            relay_managers = RelayManager(timeout=relaytimeout)
            relay_managers.add_relay_list(rl)

            filters = FiltersList([Filters(ids=[sourceid], limit=5)])
            subscription_id = uuid.uuid1().hex
            relay_managers.add_subscription_on_all_relays(subscription_id, filters)
            relay_managers.run_sync()

            event_msg = relay_managers.message_pool.get_event()
            relay_managers.close_all_relay_connections()
            text = event_msg.event.content
            request_form["optStr"] = 'text=' + text + ';translation_lang=' + translation_lang

    elif task == "summarization":
        request_form["mode"] = "PREDICT_STATIC"
        print("[Nostr] Not supported yet")
        # call OpenAI API or use a local LLM
        # add length variableF

    elif task == "image-to-image":
        request_form["mode"] = "PREDICT_STATIC"

        if event.get_tag_list('i')[0][1] == "url":
            url = event.get_tag_list('i')[0][0]

        prompt = ""
        negative_prompt = ""
        strength = 0.75
        guidance_scale = 7.5
        params = event.get_tag_list('params')
        for param in params:
            if param[0] == "prompt":  # check for paramtype
                prompt = param[1]
            elif param[0] == "negative_prompt":  # check for paramtype
                negative_prompt = param[1]
            elif param[0] == "strength":  # check for paramtype
                strength = float(param[1])
            elif param[0] == "guidance_scale":  # check for paramtype
                guidance_scale = float(param[1])
        request_form["optStr"] = 'url=' + url + ';prompt=' + prompt + ';negative_prompt=' + negative_prompt + ';strength=' + strength + ';guidance_scale=' + guidance_scale



    elif task == "text-to-image":
        request_form["mode"] = "PREDICT_STATIC"
        request_form["trainerFilePath"] = 'text-to-image'

        if event.get_tag_list('i')[0][1] == "text":
            prompt = event.get_tag_list('i')[0][0]
        elif event.get_tag_list('i')[0][1] == "event":
            sourceid = event.get_tag_list('i')[0][0]
            relay_managers = RelayManager(timeout=relaytimeout)
            relay_managers.add_relay_list(rl)

            filters = FiltersList([Filters(ids=[sourceid], limit=5)])
            subscription_id = uuid.uuid1().hex
            relay_managers.add_subscription_on_all_relays(subscription_id, filters)
            relay_managers.run_sync()

            event_msg = relay_managers.message_pool.get_event()
            relay_managers.close_all_relay_connections()
            prompt = event_msg.event.content

        negative_prompt = ""
        params = event.get_tag_list('params')
        for param in params:
            if param[0] == "prompt":  # check for paramtype
                prompt = param[1]
            elif param[0] == "negative_prompt":  # check for paramtype
                negative_prompt = param[1]

        request_form["optStr"] = 'prompt=' + prompt + ';negative_prompt=' + negative_prompt

    elif task == "image-upscale":
        # let's download an  image
        if event.get_tag_list('i')[0][1] == "url":
            url = event.get_tag_list('i')[0][0]
            request_form["optStr"] = 'url=' + url

    return request_form


def sendJobStatusReaction(originalevent, status, isPaid=True, amount=0):
    reaction = '+'

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

    privkey = PrivateKey.from_hex(os.environ["NOVA_NOSTR_KEY"])
    pubkey = privkey.public_key
    relay_managers = RelayManager(timeout=relaytimeout)

    if len(originalevent.get_tag_list("relays")) > 0:
        relaystosend = originalevent.get_tag_list("relays")[0]
        # If no relays are given, use default
    else:
        relaystosend = []

    if (len(relaystosend) == 0):
        relay_managers.add_relay_list(rl)
    # else use relays from tags
    else:
        for relay in relaystosend:
            if relay not in ignore_url_list:
                relay_managers.add_relay(relay)

    event = Event(reaction)
    event.kind = 68003
    event.add_tag('e', originalevent.id)
    event.add_tag('p', originalevent.pubkey)
    event.add_tag('status', status)

    if status == "success" or status == "failed":  #
        for x in JobstoWatch:
            if x.id == originalevent.id:
                isPaid = x.isPaid
                amount = x.amount
                break

    if status == "payment-required" or (status == "started" and not isPaid):
        JobstoWatch.append(
            JobToWatch(id=originalevent.id, timestamp=originalevent.created_at, amount=amount, isPaid=isPaid,
                       status=status, result="", isProcessed=False))
        print(str(JobstoWatch))
    if status == "payment-required" or (status == "started" and not isPaid) or (status == "success" and not isPaid):
        event.add_tag('amount', str(amount))
    event.sign(privkey.hex())

    relay_managers.publish_event(event)
    relay_managers.run_sync()
    time.sleep(3)

    relay_managers.close_all_relay_connections()
    print("[Nostr] Sent Kind 68003 Reaction: " + status + " " + str(event.to_dict()))

    return event.to_dict()


def CheckEventStatus(content, originaleventstr: str, useBot=False):
    originalevent = Event.from_dict(json.loads(originaleventstr.replace("'", "\"")))

    for x in JobstoWatch:
        if x.id == originalevent.id:
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
        print(str(JobstoWatch))
        if (useBot):
            privkey = PrivateKey.from_hex(os.environ["NOVA_NOSTR_KEY"])
            sendDM(privkey.hex(), originalevent.pubkey, "Your Result: \n\n" + content)
        else:
            sendNostrReplyEvent(content, originaleventstr)
            sendJobStatusReaction(originalevent, "success")


def sendNostrReplyEvent(content, originaleventstr):
    # Once the Job is finished we reply with the results with a 68002 event
    privkey = PrivateKey.from_hex(os.environ["NOVA_NOSTR_KEY"])
    pubkey = privkey.public_key
    originalevent = Event.from_dict(json.loads(originaleventstr.replace("'", "\"")))

    relay_managers = RelayManager(timeout=relaytimeout)
    relaystosend = []
    if len(originalevent.get_tag_list("relays")) > 0:
        relaystosend = originalevent.get_tag_list("relays")[0]
    # If no relays are given, use default
    if (len(relaystosend) == 0):
        relay_managers.add_relay_list(rl)
    # else use relays from tags
    else:
        for relay in relaystosend:
            if not relay == "wss://nostr-pub.wellorder.net":  # causes errors
                relay_managers.add_relay(relay)

    filters = FiltersList([Filters(authors=[pubkey.hex()], limit=100)])
    subscription_id = uuid.uuid1().hex
    relay_managers.add_subscription_on_all_relays(subscription_id, filters)
    if len(originalevent.get_tag_list("output")) > 0:
        if originalevent.get_tag_list("output")[0][0] == "text/plain":
            result = ""
            for name in content["name"]:
                clearedName = str(name).lstrip("\'").rstrip("\'")
                result = result + clearedName + "\n"
            content = str(result).replace("\"", "").replace('[', "").replace(']', "").lstrip(None)

    event = Event(str(content))
    event.kind = 68002
    event.add_tag('request', str(originalevent.to_dict()).replace("'", "\""))
    event.add_tag('e', originalevent.id)
    event.add_tag('p', originalevent.pubkey)
    event.add_tag('status', "success")
    event.sign(privkey.hex())

    relay_managers.publish_event(event)
    relay_managers.run_sync()
    time.sleep(5)
    relay_managers.close_all_relay_connections()

    print("[Nostr] 68002 Job Response event sent: " + str(event.to_dict()))

    return event.to_dict()


# HELPER

def savConfig(dbUser, dbPassword, dbServer, database, role, annotator):
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


def getIndexOfFirstLetter(ip):
    index = 0
    for c in ip:
        if c.isalpha():
            return index
        else:
            index = index + 1

    return len(input);


def isBlackListed(pubkey):
    # Store  lists of blaclisted npubs that can no do processing
    # blacklisting and whitelsting should be moved to a database and probably get some expiry
    blacklisted_all_tasks = []
    if any(pubkey == c for c in blacklisted_all_tasks):
        return True
    return False


def isWhiteListed(pubkey, task):
    privkey = PrivateKey.from_hex(os.environ["NOVA_NOSTR_KEY"])
    # Store a list of whistlisted npubs that can do free processing

    # Store  ists of whistlisted npubs that can do free processing for specific tasks
    pablo7z = "fa984bd7dbb282f07e16e7ae87b26a2a7b9b90b7246a44771f0cf5ae58018f52"
    dbth = "99bb5591c9116600f845107d31f9b59e2f7c7e09a1ff802e84f1d43da557ca64"
    localnostrtest = "558497db304332004e59387bc3ba1df5738eac395b0e56b45bfb2eb5400a1e39"
    # PublicKey.from_npub("npub...").hex()
    whitelsited_npubs_speechtotext = [localnostrtest, pablo7z]  # remove this to test LN Zaps
    whitelsited_npubs_translation = [localnostrtest]
    whitelsited_npubs_texttoimage = [localnostrtest]
    whitelsited_npubs_imagetoimage = [localnostrtest]
    whitelsited_npubs_imageupscale = [localnostrtest]

    whitelsited_all_tasks = [privkey.public_key.hex()]

    if (task == "speech-to-text"):
        if any(pubkey == c for c in whitelsited_npubs_speechtotext) or any(pubkey == c for c in whitelsited_all_tasks):
            return True
    elif (task == "translation"):
        if any(pubkey == c for c in whitelsited_npubs_translation) or any(pubkey == c for c in whitelsited_all_tasks):
            return True
    elif (task == "text-to-image"):
        if any(pubkey == c for c in whitelsited_npubs_texttoimage) or any(pubkey == c for c in whitelsited_all_tasks):
            return True
    elif (task == "image-to-image"):
        if any(pubkey == c for c in whitelsited_npubs_imagetoimage) or any(pubkey == c for c in whitelsited_all_tasks):
            return
    elif (task == "image-upscale"):
        if any(pubkey == c for c in whitelsited_npubs_imageupscale) or any(pubkey == c for c in whitelsited_all_tasks):
            return True
    return False
