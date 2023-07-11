import mimetypes
import os
import string

from decord import AudioReader, cpu
from pynostr.encrypted_dm import EncryptedDirectMessage
from translatepy.translators.google import GoogleTranslate
import ffmpegio
from pynostr.relay_manager import RelayManager
from pynostr.filters import FiltersList, Filters
from pynostr.event import EventKind
import json
from pynostr.event import Event
from pynostr.key import PrivateKey, PublicKey

from nova_server.utils.db_utils import add_new_session_to_db, db_entry_exists
from configparser import ConfigParser

import time
import datetime
import requests

import uuid

#TODO
# check expiry of tasks/available output format/model/ (task is checked already). if not available ignore the job,
# send reaction on error (send sats back ideally, library meh, same for under payment),
# send reaction processing-scheduled when task is waiting for previous task to finish, max limit to wait?
# store whitelist (and maybe a blacklist) in a config/db
# clear list of  tasks (IDstoWatch) to watch after some time (timeout if invoice not paid),
# consider max-sat amount at all,
# consider reactions from customers (Kind 07 event)
# consider encrypted DMS with tasks (decrypt seems broken in pynostr, or dependecy version of)
# add more output formats (webvtt, srt)
# refactor translate to own module
# add summarization task (GPT4all?, OpenAI api?) in own module
# whisper_large-v2 only works once? (base is fine)

# purge database and files from time to time?

sinceLastNostrUpdate = int((datetime.datetime.now() - datetime.timedelta(minutes=1)).timestamp())
IDstoWatch = []

def nostrReceiveAndManageNewEvents():
    #privkey = PrivateKey.from_hex(os.environ["NOVA_NOSTR_KEY"])
    global sinceLastNostrUpdate
    global IDstoWatch

    relay_manager = RelayManager(timeout=2)
    relay_manager.add_relay("wss://relay.damus.io")
    relay_manager.add_relay("wss://relay.snort.social")
    #print("[Nostr] Listen to new events since: " + str(sinceLastNostrUpdate))

    vendingFilter = Filters(kinds=[68001], since=sinceLastNostrUpdate, limit=20)
    zapFilter = Filters(kinds=[EventKind.ZAPPER], limit =20, since=sinceLastNostrUpdate)
    zapFilter.add_arbitrary_tag('e', IDstoWatch)
    #reactFilter = Filters(kinds=[EventKind.REACTION], limit=20, since=sinceLastNostrUpdate)
    #reactFilter.add_arbitrary_tag('p', privkey.public_key.hex())
    #dmFilter = Filters(kinds=[EventKind.ENCRYPTED_DIRECT_MESSAGE], limit=5, since=sinceLastNostrUpdate)
    #dmFilter.add_arbitrary_tag('p', [privkey.public_key.hex()])
    filters = FiltersList([vendingFilter, zapFilter])
    subscription_id = uuid.uuid1().hex
    relay_manager.add_subscription_on_all_relays(subscription_id, filters)
    relay_manager.run_sync()

    while relay_manager.message_pool.has_events():
        event_msg = relay_manager.message_pool.get_event()
        sinceLastNostrUpdate = int(max(event_msg.event.created_at+1, sinceLastNostrUpdate))
        event = event_msg.event
        # check for Task, for this demo use case only get active when task is speech-to-text
        if event.kind == 68001 and (event.get_tag_list('j')[0][0] == "speech-to-text" or event.get_tag_list('j')[0][0] == "translation"):
           # if npub sending the 68001 event is whitelisted, we just do the work
           if isWhitelisted(event.pubkey, event.get_tag_list('j')[0][0]):
               print("[Nostr] Whitelisted for task " + event.get_tag_list('j')[0][0] + ". Starting processing..")
               doWork(event)
           # otherwise send payment request
           else:
                if event.get_tag_list('j')[0][0] == "translation":
                    PROCESSINGCOSTPERUNIT = 2
                    duration = 1  # todo get task duration
                    print("[Nostr][Payment required] New Nostr translation Job event: " + str(event.to_dict()))
                elif event.get_tag_list('j')[0][0] == "speech-to-text":
                    PROCESSINGCOSTPERUNIT = 5
                    duration = 1  # todo get task duration
                    print("[Nostr][Payment required] New Nostr speech-to-text Job event: " + str(event.to_dict()))
                else:
                    print("[Nostr] Task " + event.get_tag_list('j')[0][0] + " is currently not supported by this instance")

                amount = PROCESSINGCOSTPERUNIT * duration * 1000 #*1000 because millisats
                if len(event.get_tag_list("bid")) > 0:
                    willingtopay = int(event.get_tag_list("bid")[0][0])
                    if willingtopay >= amount:
                        sendJobStatusReaction(event, "payment-required", willingtopay) # Take what user is willing to pay, min server rate
                    else:
                        sendJobStatusReaction(event, "payment-rejected", amount) # Reject and tell user minimum amount

                else:     # If there is no bid, just request server rate from user
                    print("[Nostr] Requesting payment for Event: " + event.id)
                    sendJobStatusReaction(event, "payment-required", amount)


        elif event.kind == EventKind.ZAPPER:
            # Zaps to us
            lninvoice = event.get_tag_list('bolt11')[0][0]
            invoicesats = ParseBolt11Invoice(lninvoice)
            print("[Nostr]Zap Received: " + str(event.to_dict()))
            eventid = event.get_tag_list("e")[0][0]

            # Get specific reaction event
            relay_manager2 = RelayManager(timeout=5)
            relay_manager2.add_relay("wss://relay.damus.io")
            relay_manager2.add_relay("wss://relay.snort.social")
            filters = FiltersList([Filters(ids=[eventid], limit=1)])
            subscription_id = uuid.uuid1().hex
            relay_manager2.add_subscription_on_all_relays(subscription_id, filters)
            relay_manager2.run_sync()
            event_msg_event7 = relay_manager2.message_pool.get_event().event
            relay_manager2.close_all_relay_connections()

            if(int(event_msg_event7.get_tag_list('amount')[0][0]) <= invoicesats*1000 ):
                print("[Nostr] Payment-request fulfilled...")
                event68001id = event_msg_event7.get_tag_list('e')[0][0]
                relay_manager3 = RelayManager(timeout=5)
                relay_manager3.add_relay("wss://relay.damus.io")
                relay_manager3.add_relay("wss://relay.snort.social")
                filters = FiltersList([Filters(ids=[event68001id], kinds=[68001], limit=1)])

                subscription_id = uuid.uuid1().hex
                relay_manager3.add_subscription_on_all_relays(subscription_id, filters)
                relay_manager3.run_sync()

                event68001 = relay_manager3.message_pool.get_event().event
                print("[Nostr] Original 68001 Job Request event found...")
                relay_manager3.close_all_relay_connections()
                sendJobStatusReaction(event68001, "payment-accepted")
                doWork(event68001)
            else:
               sendJobStatusReaction(event68001, "payment-rejected", invoicesats*1000)
               print("[Nostr] Invoice was not paid sufficiently")

        elif event.kind == EventKind.REACTION:
            print("[Nostr]Reaction Received: " + str(event.to_dict()))
    relay_manager.close_all_relay_connections()

def doWork(event68001):
    if event68001.kind == 68001:
        if event68001.get_tag_list('j')[0][0] == "speech-to-text":
            print("[Nostr] Adding Nostr speech-to-text Job event: " + str(event68001.to_dict()))
            sendJobStatusReaction(event68001, "processing-started")
            request_form = createRequestFormfromNostrEvent(event68001)
            organizeInputData(event68001, request_form)
            url = 'http://' + os.environ["NOVA_HOST"] + ':' + os.environ["NOVA_PORT"] + '/' + str(
                request_form["mode"]).lower()
            headers = {'Content-type': 'application/x-www-form-urlencoded'}
            requests.post(url, headers=headers, data=request_form)

        elif event68001.get_tag_list('j')[0][0] == "translation":
             print("[Nostr] Adding translation Job event: " + str(event68001.to_dict()))
             sendJobStatusReaction(event68001, "processing-started")
             createRequestFormfromNostrEvent(event68001) # this includes translation, should be moved to a script
        else:
            print("[Nostr] Task " + event68001.get_tag_list('j')[0][0] + " is currently not supported by this instance")

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
            ext = "mp4"
            type = "audio"

        filename = data_dir + '\\' + request_form["database"] + '\\' + session + '\\' + request_form[
                "roles"] + '.original' + type + '.' + ext

        if not os.path.exists(filename):
            file = open(filename, 'wb')
            for chunk in req.iter_content(100000):
                file.write(chunk)
            file.close()

            file_reader = AudioReader(filename, ctx=cpu(0), mono=False)
            duration = file_reader.duration()

            if float(request_form['endTime']) == 0.0:
                 end_time= duration
            elif float (request_form['endTime']) > duration:
                 end_time = duration
            else: end_time = float(request_form['endTime'])
            if(float (request_form['startTime']) < 0.0 or float(request_form['startTime']) > float(request_form['endTime'])) :
                start_time = 0.0
            else: start_time = float(request_form['startTime'])

            #for now we cut and convert all files to mp3
            finalfilename = data_dir + '\\' + request_form["database"] + '\\' + session + '\\' + request_form["roles"]+'.'+request_form["streamName"] + '.mp3'
            fs, x = ffmpegio.audio.read(filename, ss=start_time, to=end_time, sample_fmt='dbl', ac=1)
            ffmpegio.audio.write(finalfilename, fs, x)

            if not db_entry_exists(request_form, session, "name", "Sessions"):
                duration = end_time - start_time
                add_new_session_to_db(request_form, duration)

def createRequestFormfromNostrEvent(event):
    # Only call this if config is not available, adjust function to your db
    # savConfig()

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

    request_form["frameSize"] = 0
    request_form["stride"] = request_form["frameSize"]
    request_form["leftContext"] = 0
    request_form["rightContext"] = 0
    request_form["nostrEvent"] = str(event.to_dict())
    request_form["sessions"] = event.id

    # defaults might be overwritten by nostr event
    alignment = "word"
    request_form["startTime"] = 0
    request_form["endTime"] = 0

    params = event.get_tag_list('params')
    for param in params:
        if param[0] == "range":  # check for paramtype
            request_form["startTime"] = param[1]
            request_form["endTime"] = param[2]
        elif param[0] == "alignment":  # check for paramtype
            alignment = param[1]
        elif param[0] == "length":  # check for paramtype
            length = param[1]
        elif param[0] == "language":  # check for paramtype
            translation_lang = str(param[1]).split('-')[0]

    if event.get_tag_list('j')[0][0] == "speech-to-text":
        # Declare specific model type e.g. whisperx_large-v2
        if len(event.get_tag_list('j')[0]) > 1:
            model = event.get_tag_list('j')[0][1]
            modelopt = str(model).split('_')[1]
        else:
            modelopt = "base"

        request_form["mode"] = "PREDICT"
        request_form["schemeType"] = "FREE"
        request_form["scheme"] = "transcript"
        request_form["streamName"] = "audio"
        request_form["trainerFilePath"] = 'models\\trainer\\' + str(request_form["schemeType"]).lower() + '\\' + str(request_form["scheme"]) + '\\audio{audio}\\whisperx\\whisperx_transcript.trainer'
        request_form["optStr"] = 'model=' + modelopt + ';alignment_mode=' + alignment + ';batch_size=2'

    elif event.get_tag_list('j')[0][0] == "translation":
        #outsource this to its own script, ideally. This is not using the database for now, but probably should.
        inputtype = "event"
        if len(event.get_tag_list('i')[0]) > 1:
            inputtype = event.get_tag_list('i')[0][1]
        if inputtype == "event":
            sourceid = event.get_tag_list('i')[0][0]
            relay_managers = RelayManager(timeout=2)
            #relay_managers.add_relay("wss://nostr-pub.wellorder.net")
            relay_managers.add_relay("wss://relay.snort.social")
            relay_managers.add_relay("wss://relay.damus.io")

            filters = FiltersList([Filters(ids=[sourceid], limit=5)])
            subscription_id = uuid.uuid1().hex
            relay_managers.add_subscription_on_all_relays(subscription_id, filters)
            relay_managers.run_sync()

            event_msg = relay_managers.message_pool.get_event()
            relay_managers.close_all_relay_connections()
            text = event_msg.event.content
            gtranslate = GoogleTranslate()
            try:
                translated_text = str(gtranslate.translate(text, translation_lang))
                print("Translated Text: " + translated_text)
            except:
                translated_text = "An error occured"

            sendNostrReplyEvent(translated_text, request_form["nostrEvent"])


    elif event.get_tag_list('j')[0][0] == "summarization":
        print("[Nostr] Not supported yet")
        # call OpenAI API or use a local LLM
        # add length variableF

    return request_form

def sendJobStatusReaction(originalevent, status, amount = 0):

    reaction = '+'

    if status == "processing-started":
        reaction = "U+1F44D" #Thumbs up
    elif status == "processing-finished":
        reaction = 'U+1F919'  # Shaka
    elif status == "payment-required":
        reaction = 'U+1F9E1' #Orange heart
    elif status == "payment-accepted":
        reaction = 'U+1F917'   # Hugging face
    elif status == "payment-rejected":
        reaction = 'U+1F648'   # See no evil
    elif status == "processing-failed":
        reaction = 'U+1F44E'  # Thumbs down

    privkey = PrivateKey.from_hex(os.environ["NOVA_NOSTR_KEY"])
    pubkey = privkey.public_key
    relay_managers = RelayManager(timeout=6)

    if len(originalevent.get_tag_list("relays")) > 0:
        relaystosend = originalevent.get_tag_list("relays")[0]
        # If no relays are given, use default
    else:
        relaystosend = []

    if (len(relaystosend) == 0):
        #relay_managers.add_relay("wss://nostr-pub.wellorder.net")
        relay_managers.add_relay("wss://relay.damus.io")
        relay_managers.add_relay("wss://relay.snort.io")
    # else use relays from tags
    else:
        for relay in relaystosend:
            relay_managers.add_relay(relay)

    event = Event(reaction)
    event.kind = 7
    event.add_tag('e', originalevent.id)
    event.add_tag('p', originalevent.pubkey)
    event.add_tag('status', status)
    if status == "payment-required" or status == "payment-rejected":
        event.add_tag('amount', str(amount))
    event.sign(privkey.hex())

    relay_managers.publish_event(event)
    relay_managers.run_sync()
    time.sleep(3)
    #if this is a payment request, listen to events that contain the event id in the e tag to check if event is zapped
    if status == "payment-required":
        IDstoWatch.append(event.id)
    relay_managers.close_all_relay_connections()
    print("[Nostr] Sent Kind 07 Reaction: " +  status + " " +  str(event.to_dict()))

    return event.to_dict()

def sendNostrReplyEvent(content, originaleventstr):
    # Once the Job is finished we reply with the results with a 68002 event

    privkey = PrivateKey.from_hex(os.environ["NOVA_NOSTR_KEY"])
    pubkey = privkey.public_key

    originalevent = Event.from_dict(json.loads(originaleventstr.replace("'", "\"")))

    relay_managers = RelayManager(timeout=6)
    relaystosend = []
    if len(originalevent.get_tag_list("relays")) > 0:
         relaystosend = originalevent.get_tag_list("relays")[0]
    # If no relays are given, use default
    if (len(relaystosend) == 0):
        #relay_managers.add_relay("wss://nostr-pub.wellorder.net")
        relay_managers.add_relay("wss://relay.snort.social")
        relay_managers.add_relay("wss://relay.damus.io")
    # else use relays from tags
    else:
        for relay in relaystosend:
            relay_managers.add_relay(relay)

    filters = FiltersList([Filters(authors=[pubkey.hex()], limit=100)])
    subscription_id = uuid.uuid1().hex
    relay_managers.add_subscription_on_all_relays(subscription_id, filters)
    if len(originalevent.get_tag_list("o") ) > 0:
        if originalevent.get_tag_list("o")[0][0] == "text/plain":
            content = (content["name"])
            content = str(content).replace("\n", "")



    event = Event(str(content))
    event.kind = 68002
    event.add_tag('request', str(originalevent.to_dict()).replace("'", "\""))
    event.add_tag('e', originalevent.id)
    event.add_tag('p', originalevent.pubkey)
    event.add_tag('status', "success")
    if len(originalevent.get_tag_list("bid")) > 1:
        event.add_tag('amount', originalevent.get_tag_list("bid")[0][1]) # this should be the actual price.
    event.sign(privkey.hex())

    relay_managers.publish_event(event)
    relay_managers.run_sync()
    time.sleep(5)
    relay_managers.close_all_relay_connections()
    sendJobStatusReaction(originalevent, "processing-finished")
    print("[Nostr] 68002 Job Response event sent: " + str(event.to_dict()))
    return event.to_dict()



#HELPER
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
    if (identifier == 'm'): number = number * 100000000 * 0.001
    elif (identifier == 'u'): number = number * 100000000 * 0.000001
    elif (identifier == 'n'): number = number * 100000000 * 0.000000001
    elif (identifier == 'p'): number = number * 100000000 * 0.000000000001

    return int(number)
def getIndexOfFirstLetter(ip):
    index = 0
    for c in ip:
        if c.isalpha():
            return index
        else:
            index = index +1

    return len(input);


def isWhitelisted(pubkey, task):
    privkey = PrivateKey.from_hex(os.environ["NOVA_NOSTR_KEY"])
    #Store a list of whistlisted npubs that can do free processing
    whitelsited_all_tasks = [privkey.public_key.hex()]
    # Store  ists of whistlisted npubs that can do free processing for specific tasks
    pablo7z = "fa984bd7dbb282f07e16e7ae87b26a2a7b9b90b7246a44771f0cf5ae58018f52"
    localnostrtest ="828c4d2b20ae3d679f9ddad0917ff9aa4c98e16612f5b4551faf447c6ce93ed8"
    #PublicKey.from_npub("npub...").hex()
    whitelsited_npubs_speechtotext = [pablo7z] # remove this to test LN Zaps
    whitelsited_npubs_translation = [localnostrtest, pablo7z]


    if(task == "speech-to-text"):
        if any(pubkey == c for c in whitelsited_npubs_speechtotext) or any(pubkey == c for c in whitelsited_all_tasks):
            return True
    elif (task == "translation"):
        if any(pubkey == c for c in whitelsited_npubs_translation) or any(pubkey == c for c in whitelsited_all_tasks):
            return True
    return False


