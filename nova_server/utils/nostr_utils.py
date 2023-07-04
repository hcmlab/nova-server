import os

from pynostr.relay_manager import RelayManager
from pynostr.filters import FiltersList, Filters
from pynostr.event import EventKind
import json
import ssl
from pynostr.event import Event
from pynostr.message_type import ClientMessageType
from pynostr.key import PrivateKey
from nova_server.utils.key_utils import get_random_name

import time
import datetime

import uuid

def createKey():
     private_key = PrivateKey()
     public_key = private_key.public_key
     print(f"Private key: {private_key.bech32()}")
     print(f"Public key: {public_key.bech32()}")
     return private_key

def runNostrTestReceive():
    kind = 68001
    relay_manager = RelayManager(timeout=2)
    relay_manager.add_relay("wss://nostr-pub.wellorder.net")
    relay_manager.add_relay("wss://relay.damus.io")
    filters = FiltersList([Filters(kinds=[kind], limit=100)])
    subscription_id = uuid.uuid1().hex
    relay_manager.add_subscription_on_all_relays(subscription_id, filters)
    relay_manager.run_sync()

    while relay_manager.message_pool.has_events():
        event_msg = relay_manager.message_pool.get_event()
        print(event_msg.event.to_dict())
    relay_manager.close_all_relay_connections()


def runNostrTestSend():
    privkeystr = os.environ["NOVA_NOSTR_KEY"]
    if (privkeystr == ""):
        privkey = createKey()
    else:
        privkey = PrivateKey.from_hex(privkeystr)
    pubkey = privkey.public_key

    relay_manager = RelayManager(timeout=6)
    relay_manager.add_relay("wss://nostr-pub.wellorder.net")
    relay_manager.add_relay("wss://relay.damus.io")

    filters = FiltersList([Filters(authors=[pubkey.hex()], limit=100)])
    subscription_id = uuid.uuid1().hex
    relay_manager.add_subscription_on_all_relays(subscription_id, filters)

    # Test Payload
    randomjobid = get_random_name()
    # job expires in 48 hours
    expiration = int(time.time()) + (60 * 60 * 48)  # 48h in seconds
    event = Event("New AI Processing JobID: " + randomjobid)
    event.kind = 68001
    event.add_tag('j', ["speech-to-text", "whisper-tiny"])
    event.add_tag('input', ["https://www.fit.vutbr.cz/~motlicek/sympatex/f2bjrop1.0.wav", "url"])
    event.add_tag('bid', ["10", "1000"])
    #event.add_tag('expiration', expiration) //Bug: Something wrong with the time, but optional
    #event.add_tag('p', pubkey.hex()) //Bug: Something wrong, but optional
    event.add_tag('d', randomjobid)
    event.sign(privkey.hex())


    relay_manager.publish_event(event)
    relay_manager.run_sync()
    time.sleep(5)  # allow the messages to send
    #while relay_manager.message_pool.has_ok_notices():
    #    ok_msg = relay_manager.message_pool.get_ok_notice()
    #    print(ok_msg)
    #while relay_manager.message_pool.has_events():
    #    event_msg = relay_manager.message_pool.get_event()
    #    print(event_msg.event.to_dict())