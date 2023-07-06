import mimetypes
import os
import pathlib

import ffmpegio
from decord import AudioReader, cpu
import ffmpegio
from numpy import double
from pynostr.message_pool import EventMessage
from pynostr.relay_manager import RelayManager
from pynostr.filters import FiltersList, Filters
from pynostr.event import EventKind
import json
import ssl
from pynostr.event import Event
from pynostr.message_type import ClientMessageType
from pynostr.key import PrivateKey
from pynostr.encrypted_dm import EncryptedDirectMessage

from nova_server.utils import log_utils
from nova_server.utils.key_utils import get_random_name, get_key_from_request_form
from nova_server.utils.db_utils import add_new_session_to_db, db_entry_exists
import hcai_datasets.hcai_nova_dynamic.utils.nova_data_utils
from configparser import ConfigParser

import time
import datetime
import requests

import uuid

sinceLastNostrUpdate = 0


def createKey():
    private_key = PrivateKey()
    public_key = private_key.public_key
    print(f"Private key: {private_key.bech32()}")
    print(f"Public key: {public_key.bech32()}")
    return private_key


def nostrReceiveAndManageNewEvents():
    privkey = PrivateKey.from_hex(os.environ["NOVA_NOSTR_KEY"])

    relay_manager = RelayManager(timeout=2)
    relay_manager.add_relay("wss://nostr-pub.wellorder.net")
    relay_manager.add_relay("wss://relay.damus.io")

    global sinceLastNostrUpdate
    sinceLastNostrUpdate = max(sinceLastNostrUpdate + 1,
                               (datetime.datetime.now() - datetime.timedelta(minutes=1)).timestamp())
    filters = FiltersList([Filters(kinds=[68001, 68002], since=sinceLastNostrUpdate, limit=5)])
    subscription_id = uuid.uuid1().hex
    relay_manager.add_subscription_on_all_relays(subscription_id, filters)
    relay_manager.run_sync()

    while relay_manager.message_pool.has_events():
        event_msg = relay_manager.message_pool.get_event()
        sinceLastNostrUpdate = max(event_msg.event.created_at, sinceLastNostrUpdate)
        event = event_msg.event
        # Attempt to work with private events, something is stil wrong with decryption
        # if event.kind == 4:
        #     dm = EncryptedDirectMessage.from_event(event)
        #     #dm.decrypt(privkey.hex(), public_key_hex=privkey.public_key.hex())
        #     dm.decrypt (privkey.hex(),
        #            recipient_pubkey=privkey.public_key.hex()
        #            )
        #     dm_event = dm.cleartext_content
        #    event = Event.from_dict(dm.cleartext_content)

        # check for Task, for this demo use case only get active when task is speech-to-text
        if event.kind == 68001 and event.get_tag_list('j')[0][0] == "speech-to-text":
            print("New Nostr Job event: " + str(event.to_dict()))
            request_form = createRequestFormfromNostrEvent(event)
            organizeInputData(event, request_form)
            url = 'http://' + os.environ["NOVA_HOST"] + ':' + os.environ["NOVA_PORT"] + '/' + str(
                request_form["mode"]).lower()
            headers = {'Content-type': 'application/x-www-form-urlencoded'}
            requests.post(url, headers=headers, data=request_form)
        elif event.kind == 68002:
            print("Nostr Job Response event: " + str(event.to_dict()))

    relay_manager.close_all_relay_connections()


def organizeInputData(event, request_form):
    data_dir = os.environ["NOVA_DATA_DIR"]
    session = event.id

    if event.get_tag_list('i')[0][1] == "url":
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

    if event.get_tag_list('j')[0][0] == "speech-to-text":
        # Declare specific model type e.g. whisperx-large
        if event.get_tag_list('j')[0][1] is not None:
            model = event.get_tag_list('j')[0][1]
            modelopt = str(model).split('-')[1]
        else:
            modelopt = "base"

        request_form["mode"] = "PREDICT"
        request_form["schemeType"] = "FREE"
        request_form["scheme"] = "transcript"
        request_form["streamName"] = "audio"
        request_form["trainerFilePath"] = 'models\\trainer\\' + str(request_form["schemeType"]).lower() + '\\' + str(request_form["scheme"]) + '\\audio{audio}\\whisperx\\whisperx_transcript.trainer'
        request_form["optStr"] = 'model=' + modelopt + ';alignment_mode=' + alignment + ';batch_size=2'

    return request_form


def sendNostrReplyEvent(anno, originaleventstr):
    # Once the Job is finished we reply with the results with a 68002 event

    privkey = PrivateKey.from_hex(os.environ["NOVA_NOSTR_KEY"])
    pubkey = privkey.public_key

    originalevent = Event.from_dict(json.loads(originaleventstr.replace("'", "\"")))

    relay_manager = RelayManager(timeout=6)
    relaystosend = originalevent.get_tag_list("relays")[0]
    # If no relays are given, use default
    if (len(relaystosend) == 0):
        relay_manager.add_relay("wss://nostr-pub.wellorder.net")
        relay_manager.add_relay("wss://relay.damus.io")
    # else use relays from tags
    else:
        for relay in relaystosend:
            relay_manager.add_relay(relay)

    filters = FiltersList([Filters(authors=[pubkey.hex()], limit=100)])
    subscription_id = uuid.uuid1().hex
    relay_manager.add_subscription_on_all_relays(subscription_id, filters)

    content = anno.data
    event = Event(str(content))
    event.kind = 68002
    event.add_tag('request', str(originalevent.to_dict()).replace("'", "\""))
    event.add_tag('e', originalevent.id)
    event.add_tag('p', originalevent.pubkey)
    event.add_tag('status', "success")
    event.add_tag('amount', originalevent.get_tag_list("bid")[0][1])
    event.sign(privkey.hex())

    relay_manager.publish_event(event)
    relay_manager.run_sync()
    time.sleep(5)
    relay_manager.close_all_relay_connections()
    return event.to_dict()


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
