"""This file contains the general logic for predicting annotations to the nova database"""
import copy
import os
import uuid
import datetime
import time
from threading import Thread

from pynostr.encrypted_dm import EncryptedDirectMessage
from pynostr.event import Event, EventKind
from pynostr.filters import FiltersList, Filters
from pynostr.key import PrivateKey, PublicKey
from pynostr.relay_manager import RelayManager


from nova_server.utils import db_utils
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
from hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable import (
    HcaiNovaDynamicIterable,
)
from nova_utils.interfaces.server_module import Trainer as iTrainer

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
            alignment="raw", task=task, rangefrom=0.0, rangeto=0, sats=1,  satsmax=10, lang="de", privatedm=False)
        return jsonify(data)


privkey = PrivateKey.from_hex("c889daba8abe65121ff16fb602fb216d4cd69ca7025f47a9455814a3ed4f9f35")
IDtoWatch = 0
def nostr_bridge_simple_test(url, expiresinminutes, alignment, task, rangefrom, rangeto, sats, satsmax, lang, privatedm = False):
    # Function that sends a default event to test, without any user inputs


    relay_manager = RelayManager(timeout=4)
    relay_manager.add_relay("wss://nostr-pub.wellorder.net")
    relay_manager.add_relay("wss://relay.damus.io")
    relay_manager.add_relay("wss://relay.snort.social")


    filters = FiltersList([Filters(authors=[privkey.public_key.hex()], limit=100)])
    subscription_id = uuid.uuid1().hex
    relay_manager.add_subscription_on_all_relays(subscription_id, filters)

    # Test Payload

    expiration = str(int((datetime.datetime.now() + datetime.timedelta(minutes=expiresinminutes)).timestamp()))
    eventToTranslateId = '2591cd2c17a786ecca79e89f8068d28206ffa11f41f06dd9539ace9d505c9ea4'
    if task == "translation":
        description = "Translate the following Event: " + eventToTranslateId + " to " + lang
    elif task == "speech-to-text":
         description = "Transcribe the attached media file"

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
    event.add_tag('relays', ["wss://nostr-pub.wellorder.net"])
    event.add_tag('bid', [str(sats*1000), str(satsmax*1000)])
    event.add_tag('exp', expiration)
    event.add_tag('p', str(privkey.public_key.hex()))
    event.sign(privkey.hex())

    global IDtoWatch
    IDtoWatch = event.id
    relay_manager.publish_event(event)
    relay_manager.run_sync()
    time.sleep(5)  # allow the messages to send
    relay_manager.close_all_relay_connections()
    return event.to_dict()




sinceLastNostrUpdate = 0

def nostclientWaitforEvents():

    global sinceLastNostrUpdate
    global IDtoWatch
    sinceLastNostrUpdate = max(sinceLastNostrUpdate + 1,
                               (datetime.datetime.now() - datetime.timedelta(minutes=1)).timestamp())
    relay_manager = RelayManager(timeout=3)
    relay_manager.add_relay("wss://nostr-pub.wellorder.net")
    # relay_manager.add_relay("wss://relay.damus.io")
    relay_manager.add_relay("wss://relay.snort.social")

    vendingFilter = Filters(kinds=[68002], since=sinceLastNostrUpdate, limit=5)
    eFilter = Filters(kinds=[EventKind.REACTION], limit=5, since=sinceLastNostrUpdate)
    eFilter.add_arbitrary_tag('e', [IDtoWatch])
    filters = FiltersList([vendingFilter, eFilter])
    subscription_id = uuid.uuid1().hex
    relay_manager.add_subscription_on_all_relays(subscription_id, filters)
    relay_manager.run_sync()

    while relay_manager.message_pool.has_events():
        event_msg = relay_manager.message_pool.get_event()
        sinceLastNostrUpdate = max(event_msg.event.created_at, sinceLastNostrUpdate)
        event = event_msg.event
        if event.kind == 68002:
            print("[Client] Nostr Job Result event: " + str(event.to_dict()))
            # todo if speech-to-text, now send translation event
        elif event.kind == EventKind.REACTION:
            # We don't need this, just to see it's there. Our own reaction
             print("[Client] Received Reaction event: " + str(event.to_dict()))
             if event.get_tag_list('amount')[0][0] is not None:
                 mSatstoSats = int(int(event.get_tag_list('amount')[0][0]) / 1000)
                 # Todo Do the Zap from here somehow (Currently need to copy the event-id to content in a client
                 print("[Client]  Send Non-private Zap with " + str(mSatstoSats) + " Sats to " + PublicKey.from_hex(
                 event.pubkey).npub + " with Content: " + event.id + " to start processing")
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
    relay_manager.add_relay("wss://nostr-pub.wellorder.net")
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
    event.add_tag('relays', ["wss://nostr-pub.wellorder.net", "wss://relay.damus.io"])
    event.add_tag('bid', ["1000", "10000"])
    event.add_tag('exp', expiration)
    event.add_tag('p', str(pubkey.hex()))
    event.sign(privkey.hex())

    relay_manager.publish_event(event)
    relay_manager.run_sync()
    time.sleep(5)  # allow the messages to send

    relay_manager.close_all_relay_connections()
    logger.info("Nostr Event sent")