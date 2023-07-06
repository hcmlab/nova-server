"""This file contains the general logic for predicting annotations to the nova database"""
import copy
import os
import uuid
import datetime
import time

from pynostr.encrypted_dm import EncryptedDirectMessage
from pynostr.event import Event
from pynostr.filters import FiltersList, Filters
from pynostr.key import PrivateKey
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
        # url = "https://www.fit.vutbr.cz/~motlicek/sympatex/f2bjrop1.0.wav"
        url = 'https://nostr.build/p/nb12277.mov'
        # url = 'https://files.catbox.moe/voxrao.wav'
        data = nostr_bridge_simple_test(
            url=url, expiresinminutes=60,
            alignment="raw", rangefrom=0.0, rangeto=20.0, sats=1,  satsmax=10, privatedm=False)
        return jsonify(data)



def nostr_bridge_simple_test(url, expiresinminutes, alignment, rangefrom, rangeto, sats, satsmax, privatedm = False):
    # Function that sends a default event to test, without any user inputs
    privkey = PrivateKey.from_hex(os.environ["NOVA_NOSTR_KEY"])
    pubkey = privkey.public_key

    relay_manager = RelayManager(timeout=6)
    relay_manager.add_relay("wss://nostr-pub.wellorder.net")
    relay_manager.add_relay("wss://relay.damus.io")

    filters = FiltersList([Filters(authors=[pubkey.hex()], limit=100)])
    subscription_id = uuid.uuid1().hex
    relay_manager.add_subscription_on_all_relays(subscription_id, filters)

    # Test Payload

    expiration = str(int((datetime.datetime.now() + datetime.timedelta(minutes=expiresinminutes)).timestamp()))
    event = Event("Transcribe the attached audio file")
    event.kind = 68001
    event.add_tag('j', ["speech-to-text", "whisperx-base"])
   # if(rangeto == 0): rangeto = currentsession.duration
    event.add_tag('params', ["range", str(rangefrom), str(rangeto)]) #this is currently not working on nova server
    event.add_tag('params', ["alignment", alignment]) #segment, word
    event.add_tag('i', [url, "url"])
    event.add_tag('relays', ["wss://nostr-pub.wellorder.net"])
    event.add_tag('bid', [str(sats*1000), str(satsmax*1000)])
    event.add_tag('exp', expiration)
    event.add_tag('p', str(pubkey.hex()))
    event.sign(privkey.hex())

    # receiving does not work yet
    if privatedm:
        recipient_pubkey = privkey.public_key #sending to self here for testing purposes
        dm = EncryptedDirectMessage()
        dm.encrypt(privkey.hex(), recipient_pubkey=recipient_pubkey.hex(), cleartext_content=str(event.to_dict()))
        dm_event = dm.to_event()
        dm_event.sign(privkey.hex())
        relay_manager.publish_event(dm_event)

    else:
        relay_manager.publish_event(event)

    relay_manager.run_sync()
    time.sleep(5)  # allow the messages to send
    relay_manager.close_all_relay_connections()
    return event.to_dict()


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