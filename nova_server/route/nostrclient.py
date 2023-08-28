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

from nostr_sdk import Keys, Client, Tag, Event, EventBuilder, Filter, HandleNotification, Timestamp, nip04_decrypt, EventId, init_logger, LogLevel


from flask import Blueprint, request, jsonify

from nova_server.utils.nostr_dvm import send_event
from nova_server.utils.thread_utils import THREADS

from nova_server.utils.key_utils import get_key_from_request_form, get_random_name
from nova_server.utils import (
    thread_utils,
    status_utils,
    log_utils,
    dataset_utils,
    import_utils,
)


nostr = Blueprint("nostr", __name__)

#TODO HINT: Only use this path with a whitelisted privkey, as zapping events is not implemented in the lib/code

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
        url = request.args.get('url', default = 'https://nostr.build/p/nb12277.mov', type = str)
        rangefrom = request.args.get('from', default=0.0, type=float)
        rangeto = request.args.get('to', default=0.0, type=float)
        alignment = request.args.get('alignment', default="raw", type=str)
        data = nostr_client_simple_test(
            url=url, expiresinminutes=60, alignment=alignment, rangefrom=rangefrom, rangeto=rangeto, sats=10, satsmax=10,)
        return jsonify(data)

def nostr_client_simple_test(url, expiresinminutes, alignment,rangefrom, rangeto, sats, satsmax):
    # Function that sends a default event to test, without any user inputs
    keys = Keys.from_sk_str("c889daba8abe65121ff16fb602fb216d4cd69ca7025f47a9455814a3ed4f9f35")
    print(f"Nostr Client public key: {keys.public_key().to_hex()}")
    expiration = str(int((datetime.datetime.now() + datetime.timedelta(minutes=expiresinminutes)).timestamp()))
    iTag = Tag.parse(["i", url, "url"])
    oTag = Tag.parse(["output", "text/plain"])
    paramTag1 = Tag.parse(["param", "model", "large-v2"])
    paramTag2 = Tag.parse(["param", "range", str(rangefrom), str(rangeto)])
    paramTag3 = Tag.parse(["alignment", alignment]) # segment, word, raw
    expTag = Tag.parse(["exp", expiration])
    bidTag = Tag.parse(['bid', str(sats * 1000), str(satsmax * 1000)])
    relaysTag = Tag.parse(['relays', "wss://relay.damus.io", "wss://blastr.f7z.xyz", "wss://relayable.org", "wss://nostr-pub.wellorder.net"])
    alttag = Tag.parse(["alt", "This is a NIP90 DVM AI task to transcribe speech to text"])
    event = EventBuilder(65002, str("Transcribe the attached file."), [iTag, oTag, paramTag1, paramTag2, paramTag3, expTag, bidTag, relaysTag, alttag]).to_event(keys)
    send_event(event)
    return event.as_json()

def nostr_client():
    relay_list = ["wss://relay.damus.io", "wss://blastr.f7z.xyz", "wss://relayable.org",
                  "wss://nostr-pub.wellorder.net"]

    keys = Keys.from_sk_str("c889daba8abe65121ff16fb602fb216d4cd69ca7025f47a9455814a3ed4f9f35")
    sk = keys.secret_key()
    pk = keys.public_key()
    print(f"Nostr Client public key: {pk.to_bech32()}")
    client = Client(keys)
    for relay in relay_list:
        client.add_relay(relay)
    client.connect()

    dmzapfilter = Filter().pubkey(pk).kinds([4, 9735]).since(Timestamp.now())  # events to us specific
    dvmfilter = (Filter().kinds([65000, 65001]).since(Timestamp.now()))  # public events
    client.subscribe([dmzapfilter, dvmfilter])

    class NotificationHandler(HandleNotification):
        def handle(self, relay_url, event):
            print(f"Received new event from {relay_url}: {event.as_json()}")
            if event.kind() == 65000:
                print("[Nostr Client]: " + event.as_json())
            elif event.kind() == 65001:
                print("[Nostr Client]: " + event.as_json())
                print("[Nostr Client]: " + event.content())

            elif event.kind() == 4:
                dec_text = nip04_decrypt(sk, event.pubkey(), event.content())
                print("[Nostr Client]: " + f"Received new msg: {dec_text}")

            elif event.kind() == 9735:
                print("[Nostr Client]: " + f"Received new zap:")
                print(event.as_json())


        def handle_msg(self, relay_url, msg):
            None

    client.handle_notifications(NotificationHandler())
    while True:
        time.sleep(5.0)

@thread_utils.ml_thread_wrapper
#not used for now
def nostr_bridge(request_form):
  print("not implemented")


if __name__ == '__main__':
    nostr_client()