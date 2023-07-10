"""This file contains the general logic for predicting annotations to the nova database"""
import base64
import copy
import json
import os
import uuid
import datetime
import time
from threading import Thread

import datetime as datetime



from pynostr.event import Event, EventKind
from pynostr.filters import FiltersList, Filters
from pynostr.key import PrivateKey, PublicKey
from pynostr.relay_manager import RelayManager

#from nostr_sdk import Keys, Client, EventBuilder, nip04_decrypt
# alternative SDK with rust bindings, maybe use this in the future

from flask import Blueprint, request, jsonify
from nova_server.utils.thread_utils import THREADS

from nova_server.utils.key_utils import get_key_from_request_form, get_random_name
from nova_server.utils import (
    thread_utils,
    status_utils,
    log_utils,
    dataset_utils,
    import_utils,
    nostr_utils,
)


nostr = Blueprint("nostr", __name__)



@nostr.route("/nostr", methods=['POST', 'GET'])
def nostr_bridge_thread():
    if request.method == "POST":
        request_form = request.form.to_dict()
        key = get_key_from_request_form(request_form)
        thread = nostr_bridge(request_form)
        status_utils.add_new_job(key, request_form=request_form)
        data = {"success": "true"}
        thread.start()
        THREADS[key] = thread
        return jsonify(data)
    elif request.method == "GET":
        task =  "speech-to-text" # "translation"
        # url = "https://www.fit.vutbr.cz/~motlicek/sympatex/f2bjrop1.0.wav"
        url = 'https://nostr.build/p/nb12277.mov'
        # url = 'https://files.catbox.moe/voxrao.wav'
        data = nostr_bridge_simple_test(
            url=url, expiresinminutes=60,
            alignment="raw", task=task, rangefrom=0.0, rangeto=0.0, sats=1, satsmax=10, eventToTranslateId = '2591cd2c17a786ecca79e89f8068d28206ffa11f41f06dd9539ace9d505c9ea4', lang="de", privatedm=False)
        return jsonify(data)


privkey = PrivateKey.from_hex("c889daba8abe65121ff16fb602fb216d4cd69ca7025f47a9455814a3ed4f9f35")
IDtoWatch = 0
def nostr_bridge_simple_test(url, expiresinminutes, alignment, eventToTranslateId, task, rangefrom, rangeto, sats, satsmax, lang, privatedm = False):
    # Function that sends a default event to test, without any user inputs


    relay_manager = RelayManager(timeout=4)
    #relay_manager.add_relay("wss://nostr-pub.wellorder.net")
    relay_manager.add_relay("wss://relay.damus.io")
    relay_manager.add_relay("wss://relay.snort.social")


    filters = FiltersList([Filters(authors=[privkey.public_key.hex()], limit=100)])
    subscription_id = uuid.uuid1().hex
    relay_manager.add_subscription_on_all_relays(subscription_id, filters)

    # Test Payload

    expiration = str(int((datetime.datetime.now() + datetime.timedelta(minutes=expiresinminutes)).timestamp()))
    someidentifier = get_random_name()
    if task == "translation":
        description = "Translate the following Event: " + eventToTranslateId + " to " + lang
    elif task == "speech-to-text":
         description = "Transcribe the attached file. ID: " + someidentifier

    event = Event(description)
    event.kind = 68001
    if task == "translation":

        #translate a previously transcribed event
        event.add_tag('j', [task])
        event.add_tag('params', ["language", lang])  # segment, word
        event.add_tag('i', [eventToTranslateId, "event"])
    elif task == "speech-to-text":
        event.add_tag('j', [task, "whisperx_base"])
        event.add_tag('params', ["range", str(rangefrom), str(rangeto)])
        event.add_tag('params', ["alignment", alignment])  # segment, word, raw
        event.add_tag('i', [url, "url"])
    event.add_tag('relays', ["wss://relay.damus.io", "wss://relay.snort.social"])
    event.add_tag('bid', [str(sats*1000), str(satsmax*1000)])
    event.add_tag('exp', expiration)
    event.add_tag('p', str(privkey.public_key.hex()))
    event.sign(privkey.hex())

    global IDtoWatch
    IDtoWatch = event.id
    relay_manager.publish_event(event)
    relay_manager.run_sync()
    print("[Nostr Client] 68001 Event Created at: " + str(event.created_at))
    print("[Nostr Client] 68001 Event Sender Pubkey: " + privkey.public_key.hex())
    time.sleep(5)  # allow the messages to send
    relay_manager.close_all_relay_connections()
    return event.to_dict()




sinceLastNostrUpdateClient = int((datetime.datetime.now() - datetime.timedelta(minutes=1)).timestamp())

def nostclientWaitforEvents():

    global sinceLastNostrUpdateClient
    global IDtoWatch
    relay_manager = RelayManager(timeout=3)
    #relay_manager.add_relay("wss://nostr-pub.wellorder.net")
    relay_manager.add_relay("wss://relay.damus.io")
    relay_manager.add_relay("wss://relay.snort.social")

    vendingFilter = Filters(kinds=[68002], since=sinceLastNostrUpdateClient, limit=5)
    eFilter = Filters(kinds=[EventKind.REACTION], limit=5, since=sinceLastNostrUpdateClient)
    eFilter.add_arbitrary_tag('e', [IDtoWatch])
    dmFilter = Filters(kinds=[EventKind.ENCRYPTED_DIRECT_MESSAGE], limit=5, since=sinceLastNostrUpdateClient)
    dmFilter.add_arbitrary_tag('p', [privkey.public_key.hex()])
    filters = FiltersList([vendingFilter, eFilter, dmFilter])
    subscription_id = uuid.uuid1().hex
    relay_manager.add_subscription_on_all_relays(subscription_id, filters)
    relay_manager.run_sync()


    while relay_manager.message_pool.has_events():
        event_msg = relay_manager.message_pool.get_event()
        sinceLastNostrUpdateClient  = int(max(event_msg.event.created_at+1, sinceLastNostrUpdateClient))
        event = event_msg.event
        #The final Result, + we add a follow up translation request in this example
        if event.kind == 68002:
            print("[Nostr Client] Nostr Job Result event: " + str(event.to_dict()))

            request = event.get_tag_list('request')[0][0]
            requestevent = Event.from_dict(json.loads(request.replace("'", "\"")))

            #just a demo use case, follow up with new event after finished with text-to-speech
            if requestevent.get_tag_list('j')[0][0] == "speech-to-text":
                lang = "es"
                print("[Nostr Client] Start follow up Job, translate transcribed media to " + lang)
                nostr_bridge_simple_test(
                    url="", expiresinminutes=60,
                    alignment="raw", task="translation", rangefrom=0.0, rangeto=0, sats=1, satsmax=10,
                    eventToTranslateId=event.id, lang=lang,
                    privatedm=False)
        elif event.kind == EventKind.REACTION:
             # Server might request payment before doing the job.
             print("[Nostr Client] Received Reaction event: " + str(event.to_dict()))
             if len(event.get_tag_list('amount')) > 0:
                 mSatstoSats = int(int(event.get_tag_list('amount')[0][0]) / 1000)
                 # Todo Do the Zap from here somehow (Currently need to copy the event-id to content in a client
                 print("[Nostr Client] Send Non-private Zap with " + str(mSatstoSats) + " Sats to " + PublicKey.from_hex(
                 event.pubkey).npub + " with Content: " + event.id + " to start processing")
        elif event.kind == EventKind.ENCRYPTED_DIRECT_MESSAGE:
            #try:
                #msg = nip04_decrypt(privkey.hex(), event.pubkey(), event.content())
                #print(f"Received new msg: {msg}")
                #event = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(), f"Echo: {msg}").to_event(keys)
                #client.send_event(event)
            #except Exception as e:
             #   print(f"Error during content decryption: {e}")

        #dm = EncryptedDirectMessage.from_event(event)
            #dm.decrypt(event.pubkey, public_key_hex=privkey.public_key.hex())
            #print(f"New dm received:{event.date_time()} {dm.cleartext_content}")
            print("[Nostr Client] Received DM(s), but this python lib can't decrypt it :(. For instructions see Reaction block above.")
    relay_manager.close_all_relay_connections()



@thread_utils.ml_thread_wrapper
#not used for now
def nostr_bridge(request_form):
    # TODO make an event from the request_form (right now this is the same at the get/test method)
    key = get_key_from_request_form(request_form)
    logger = log_utils.get_logger_for_thread(key)

    log_conform_request = dict(request_form)
    log_conform_request["password"] = "---"

    logger.info("Action 'Nostr Sending bridge' started.")
    privkey = PrivateKey.from_hex(os.environ["NOVA_NOSTR_KEY"])
    pubkey = privkey.public_key

    relay_manager = RelayManager(timeout=6)
    #relay_manager.add_relay("wss://nostr-pub.wellorder.net")
    relay_manager.add_relay("wss://relay.snort.social")
    relay_manager.add_relay("wss://relay.damus.io")

    filters = FiltersList([Filters(authors=[pubkey.hex()], limit=100)])
    subscription_id = uuid.uuid1().hex
    relay_manager.add_subscription_on_all_relays(subscription_id, filters)

    # Test Payload
    randomjobid = get_random_name()
    # job expires in 1 hour
    expiration = str(int((datetime.datetime.now() + datetime.timedelta(minutes=60)).timestamp()))
    event = Event("New AI Processing JobID: " + randomjobid)
    event.kind = 68001
    event.add_tag('j', ["speech-to-text", "whisperx-base"])
    event.add_tag('params', ["range", "0", "5"])
    event.add_tag('i', ["https://www.fit.vutbr.cz/~motlicek/sympatex/f2bjrop1.0.wav", "url"])
    event.add_tag('relays', ["wss://relay.damus.io", "wss://relay.snort.social"])
    event.add_tag('bid', ["1000", "10000"])
    event.add_tag('exp', expiration)
    event.add_tag('p', str(pubkey.hex()))
    event.sign(privkey.hex())

    relay_manager.publish_event(event)
    relay_manager.run_sync()
    time.sleep(5)  # allow the messages to send

    relay_manager.close_all_relay_connections()
    logger.info("Nostr Event sent")