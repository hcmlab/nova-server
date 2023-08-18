import json

import os
import re
import urllib
from dataclasses import dataclass
from datetime import timedelta
from sqlite3 import Error
from urllib.parse import urlparse

import pandas as pd
from bech32 import bech32_decode, convertbits

import requests
import emoji
import ffmpegio

from Crypto.Cipher import AES

from decord import AudioReader, cpu
from nostr_sdk import PublicKey, Keys, Client, Tag, Event, EventBuilder, Filter, HandleNotification, Timestamp, \
    nip04_decrypt, EventId, Metadata, nostr_sdk
import time

from nova_utils.ssi_utils.ssi_anno_utils import Anno

from nova_server.route.predict_static import LLAMA2
from nova_server.utils.db_utils import db_entry_exists, add_new_session_to_db
from nova_server.utils.mediasource_utils import download_podcast, downloadYouTube
from nova_server.utils.dvm_config import DVMConfig
from configparser import ConfigParser
import sqlite3


# TODO
# check expiry of tasks/available output format/model/ (task is checked already). if not available ignore the job,
# send reaction on error (send sats back ideally, find out with lib, same for under payment),
# send reaction processing-scheduled when task is waiting for previous task to finish, max limit to wait?
# consider reactions from customers (Kind 65000 event)
# add more output formats (webvtt, srt)
# purge database and files from time to time?
# Show preview of longer transcriptions, then ask for zap




@dataclass
class JobToWatch:
    event_id: str
    timestamp: int
    is_paid: bool
    amount: int
    status: str
    result: str
    is_processed: bool
    bolt11: str
    payment_hash: str
    expires: int
    from_bot: bool


job_list = []


# init_logger(LogLevel.DEBUG)

def nostr_server():
    global job_list
    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    sk = keys.secret_key()
    pk = keys.public_key()
    print(f"Nostr Bot/DVM public key: {pk.to_bech32()}")
    client = Client(keys)
    for relay in DVMConfig.RELAY_LIST:
        client.add_relay(relay)
    client.connect()
    dm_zap_filter = Filter().pubkey(pk).kinds([4, 9734, 9735]).since(Timestamp.now())
    dvm_filter = (Filter().kinds([66000, 65002, 65003, 65004, 65005]).since(Timestamp.now()))
    client.subscribe([dm_zap_filter, dvm_filter])

    create_sql_table()
    # clear_db()
    admin_make_database_updates()

    class NotificationHandler(HandleNotification):
        def handle(self, relay_url, event):
            if 65002 <= event.kind() <= 66000:
                user = get_or_add_user(event.pubkey().to_hex())
                print(f"[Nostr] Received new NIP90 Job Request from {relay_url}: {event.as_json()}")
                is_whitelisted = user[2]
                is_blacklisted = user[3]
                if is_blacklisted:
                    send_job_status_reaction(event, "error", client=client)
                    print("[Nostr] Request by blacklisted user, skipped")
                elif check_task_is_supported(event):
                    task = get_task(event)
                    amount = get_amount_per_task(task)
                    if amount is None:
                        return

                    if is_whitelisted or task == "chat":
                        print("[Nostr] Whitelisted for task " + task + ". Starting processing..")
                        send_job_status_reaction(event, "processing", True, 0, client=client)
                        do_work(event, is_from_bot=False)
                    # otherwise send payment request
                    else:
                        bid = 0
                        for tag in event.tags():
                            if tag.as_vec()[0] == 'bid':
                                bid = int(tag.as_vec()[1])

                        print("[Nostr][Payment required] New Nostr " + task + " Job event: " + event.as_json())
                        if bid > 0:
                            bid_offer = int(bid / 1000)
                            if (bid_offer > DVMConfig.AUTOPROCESS_MIN_AMOUNT or
                                    bid_offer < DVMConfig.AUTOPROCESS_MAX_AMOUNT):
                                print("[Nostr][Auto-processing: Payment suspended to end] Job event: " + str(
                                    event.as_json()))
                                do_work(event, is_from_bot=False)
                            else:
                                if bid_offer >= amount:
                                    send_job_status_reaction(event, "payment-required", False,
                                                             bid_offer,
                                                             client=client)
                                else:
                                    send_job_status_reaction(event, "payment-rejected", False,
                                                             amount,
                                                             client=client)  # Reject and tell user minimum amount

                        else:  # If there is no bid, just request server rate from user
                            print("[Nostr] Requesting payment for Event: " + event.id().to_hex())
                            send_job_status_reaction(event, "payment-required",
                                                     False, amount, client=client)

            elif event.kind() == 4:
                sender = event.pubkey().to_hex()
                try:
                    dec_text = nip04_decrypt(sk, event.pubkey(), event.content())
                    user = get_or_add_user(sender)
                    nip05 = user[4]
                    name = user[6]
                    # Get nip05,lud16 and name from profile and store them in db.
                    if nip05 == "" or nip05 is None:
                        try:
                            profile_filter = Filter().kind(0).author(event.pubkey().to_hex()).limit(1)
                            events = client.get_events_of([profile_filter], timedelta(seconds=3))
                            if len(events) > 0:
                                ev = events[0]
                                metadata = Metadata.from_json(ev.content())
                                name = metadata.get_display_name()
                                if name == "" or name is None:
                                    name = metadata.get_name()
                                nip05 = metadata.get_nip05()
                                lud16 = metadata.get_lud16()
                                update_sql_table(user[0], user[1], user[2], user[3], nip05, lud16, name,
                                                 Timestamp.now().as_secs())
                                user = get_from_sql_table(user[0])
                                if nip05 == "" or nip05 is None:

                                    if DVMConfig.REQUIRES_NIP05 and user[1] <= DVMConfig.NEW_USER_BALANCE:
                                        time.sleep(1.0)
                                        message = (("In order to reduce misuse by bots, a NIP05 address or a balance "
                                                    "higher than the free credits (") + str(DVMConfig.NEW_USER_BALANCE)
                                                   + " Sats) is required to use this service. You can zap any of my "
                                                     "notes or my profile using public or private zaps. "
                                                     "Zapplepay is also supported")

                                        evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(), message,
                                                                                    event.id()).to_event(keys)
                                        send_event(evt, client)
                                        return
                        except Exception as e:
                            print(e)

                    # upate last active status
                    update_sql_table(user[0], user[1], user[2], user[3], user[4], user[5], user[6],
                                     Timestamp.now().as_secs())
                    if any(dec_text.startswith("-" + s) for s in DVMConfig.SUPPORTED_TASKS):
                        task = str(dec_text).split(' ')[0].removeprefix('-')
                        print("Request from " + name + " (" + nip05 + ") Task: " + task)
                        required_amount = get_amount_per_task(task)
                        balance = user[1]
                        is_whitelisted = user[2]
                        is_blacklisted = user[3]
                        if is_blacklisted:
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(),
                                                                "Your are currently blocked from all services.",
                                                                None).to_event(keys)
                            send_event(evt, client)
                        elif is_whitelisted or balance >= required_amount:
                            time.sleep(3.0)
                            if not is_whitelisted:
                                balance = max(balance - required_amount, 0)
                                update_sql_table(sender, balance, is_whitelisted, is_blacklisted, user[4], user[5],
                                                 user[6], Timestamp.now().as_secs())
                                evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(),
                                      "Your Job is now scheduled. New balance is " + str(balance)
                                      + " Sats.\nI will DM you once I'm done processing.", event.id()).to_event(keys)
                            else:
                                evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(),
                                      "Your Job is now scheduled. As you are whitelisted, your balance remains at "
                                      + str(balance) + " Sats.\nI will DM you once I'm done processing.",
                                      event.id()).to_event(keys)

                            send_event(evt, client)
                            tags = parse_bot_command_to_event(dec_text)
                            for tag in tags:
                                if tag.as_vec()[0] == "j":
                                    task = tag.as_vec()[1]

                            tags.append(Tag.parse(["p", event.pubkey().to_hex()]))
                            evt = EventBuilder(4, "", tags).to_event(keys)

                            expires = event.created_at().as_secs() + (60 * 60)
                            job_list.append(
                                JobToWatch(event_id=evt.id().to_hex(), timestamp=event.created_at().as_secs(),
                                           amount=required_amount, is_paid=True, status="processing", result="",
                                           is_processed=False, bolt11="", payment_hash="", expires=expires, from_bot=True))
                            do_work(evt, is_from_bot=True)

                        else:
                            print("payment-required")
                            time.sleep(3.0)
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(),
                                "Balance required, please zap this note with at least " + str(required_amount)
                                + " Sats to start directly, or zap me that amount elsewhere, then try again.",
                                event.id()).to_event(keys)
                            expires = event.created_at().as_secs() + (60 * 60)
                            job_list.append(
                                JobToWatch(event_id=evt.id().to_hex(), timestamp=event.created_at().as_secs(),
                                           amount=required_amount, is_paid=False, status="payment-required", result="",
                                           is_processed=False, bolt11="", payment_hash="", expires=expires, from_bot=True))
                            send_event(evt, client)

                    elif not DVMConfig.PASSIVE_MODE:
                        print("Request from " + name + " (" + nip05 + ") Message: " + dec_text)
                        if str(dec_text).startswith("-balance"):
                            get_or_add_user(sender)
                            balance = user[1]
                            time.sleep(3.0)
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(),
                                "Your current balance is " + str(balance) + " Sats. Zap me to add to your balance. "
                                "I support both public and private Zaps, as well as Zapplepay.",
                                 None).to_event(keys)
                            send_event(evt, client)
                        elif str(dec_text).startswith("-help") or str(dec_text).startswith("- help") or str(
                                dec_text).startswith("help"):
                            time.sleep(3.0)
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(), get_bot_help_text(),
                                                                        event.id()).to_event(keys)
                            send_event(evt, client)
                        elif str(dec_text).lower().__contains__("bitcoin"):
                            time.sleep(3.0)
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(),
                                 "#Bitcoin? There is no second best.\n\nhttps://cdn.nostr.build/p/mYLv.mp4",
                                  event.id()).to_event(keys)
                            send_event(evt, client)
                        elif not str(dec_text).startswith("-"):
                            # Contect LLAMA Server in parallel to cue.
                            answer = LLAMA2(dec_text, event.pubkey().to_hex())
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(), answer,
                                                                        event.id()).to_event(keys)
                            send_event(evt, client)

                except Exception as e:
                    print(f"Error during content decryption: {e}")
            elif event.kind() == 9734:
                print(event.as_json())
            elif event.kind() == 9735:
                zapped_event = None
                invoice_amount = 0
                anon = False
                sender = event.pubkey()
                try:
                    for tag in event.tags():
                        if tag.as_vec()[0] == 'bolt11':
                            invoice_amount = parse_bolt11_invoice(tag.as_vec()[1])
                        elif tag.as_vec()[0] == 'e':
                            zapped_event = get_event_by_id(tag.as_vec()[1])
                        elif tag.as_vec()[0] == 'description':
                            zap_request_event = Event.from_json(tag.as_vec()[1])
                            sender = check_for_zapplepay(zap_request_event.pubkey().to_hex(),
                                                         zap_request_event.content())
                            for ztag in zap_request_event.tags():
                                if ztag.as_vec()[0] == 'anon':
                                    if len(ztag.as_vec()) > 1:
                                        print("Private Zap received.")
                                        decrypted_content = decrypt_private_zap_message(ztag.as_vec()[1],
                                                                                        keys.secret_key(),
                                                                                        zap_request_event.pubkey())
                                        decrypted_private_event = Event.from_json(decrypted_content)
                                        if decrypted_private_event.kind() == 9733:
                                            sender = decrypted_private_event.pubkey().to_hex()
                                            message = decrypted_private_event.content()
                                            if message != "":
                                                print("Zap Message: " + message)
                                    else:
                                        anon = True
                                        print("Anonymous Zap received. Unlucky, I don't know from whom, and never will")
                    user = get_or_add_user(sender)
                    print("Zap received: " + str(invoice_amount) + " Sats from" + user[6])
                    if zapped_event is not None:
                        if zapped_event.kind() == 65000:  # if a reaction by us got zapped
                            amount = 0
                            job_event = None
                            for tag in zapped_event.tags():
                                if tag.as_vec()[0] == 'amount':
                                    amount = int(float(tag.as_vec()[1]) / 1000)
                                elif tag.as_vec()[0] == 'e':
                                    job_event = get_event_by_id(tag.as_vec()[1])

                            if job_event is not None and check_task_is_supported(job_event):
                                if amount <= invoice_amount:
                                    print("[Nostr] Payment-request fulfilled...")
                                    send_job_status_reaction(job_event, "processing", client=client)
                                    indices = [i for i, x in enumerate(job_list) if x.id == job_event.id().to_hex()]
                                    index = -1
                                    if len(indices) > 0:
                                        index = indices[0]
                                    if index > -1:
                                        if job_list[index].is_processed:  # If payment-required appears after processing
                                            job_list[index].is_paid = True
                                            check_event_status(job_list[index].result, str(job_event.as_json()))
                                        elif not (job_list[index]).is_processed:
                                            # If payment-required appears before processing
                                            job_list.pop(index)
                                            do_work(job_event, is_from_bot=False)
                                else:
                                    send_job_status_reaction(job_event, "payment-rejected",
                                                             False, invoice_amount, client=client)
                                    print("[Nostr] Invoice was not paid sufficiently")
                        elif zapped_event.kind() == 4 and not DVMConfig.PASSIVE_MODE:
                            required_amount = 50
                            job_event = None
                            for tag in zapped_event.tags():
                                if tag.as_vec()[0] == 'e':
                                    job_event = get_event_by_id(tag.as_vec()[1])

                            if job_event is not None:
                                indices = [i for i, x in enumerate(job_list) if x.id == zapped_event.id().to_hex()]
                                print(str(indices))
                                if len(indices) == 1:

                                    dec_text = nip04_decrypt(sk, job_event.pubkey(), job_event.content())
                                    tags = parse_bot_command_to_event(dec_text)
                                    tags.append(Tag.parse(["p", job_event.pubkey().to_hex()]))
                                    work_event = EventBuilder(4, "", tags).to_event(keys)
                                    for tag in tags:
                                        if tag.as_vec()[0] == "j":
                                            task = tag.as_vec()[1]
                                            required_amount = get_amount_per_task(task)
                                    if invoice_amount >= required_amount:
                                        dm_event = EventBuilder.new_encrypted_direct_msg(keys, job_event.pubkey(),
                                                    "Zap ⚡️ received! Your Job is now scheduled. I will DM you "
                                                    "once I'm done processing.", None).to_event(keys)
                                        send_event(dm_event, client)
                                        job_list.pop(indices[0])
                                        print(work_event.as_json())
                                        do_work(work_event, is_from_bot=True)
                                    elif not anon:
                                        update_user_balance(sender, invoice_amount)
                                elif not anon:
                                    update_user_balance(sender, invoice_amount)
                            elif not anon:
                                update_user_balance(sender, invoice_amount)
                        elif zapped_event.kind() == 65001:
                            print("Someone zapped the result of an exisiting Task. Nice")
                        elif not anon and not DVMConfig.PASSIVE_MODE:
                            update_user_balance(sender, invoice_amount)
                            # a regular note
                    elif not anon and not DVMConfig.PASSIVE_MODE:
                        update_user_balance(sender, invoice_amount)

                except Exception as e:
                    print(f"Error during content decryption: {e}")

        def handle_msg(self, relay_url, msg):
            return

    # PREPARE REQUEST FORM AND DATA AND SEND TO PROCESSING
    def create_requestform_from_nostr_event(event, is_bot=False):
        task = get_task(event)

        # Read config.ini file
        config_object = ConfigParser()
        config_object.read("nostrconfig.ini")
        if len(config_object) == 1:
            db_user = input("Please enter a DB User:\n")
            db_password = input("Please enter DB User Password:\n")
            db_server = input("Please enter a DB Host:\n")
            save_config(db_user, db_password, db_server, "nostr_test", "nostr", "system")
            config_object.read("nostrconfig.ini")

        user_info = config_object["USERINFO"]
        server_config = config_object["SERVERCONFIG"]

        request_form = {"dbServer": server_config["dbServer"], "dbUser": user_info["dbUser"],
                        "dbPassword": user_info["dbPassword"], "database": server_config["database"],
                        "roles": server_config["roles"], "annotator": server_config["annotator"],
                        "flattenSamples": "false", "jobID": event.id().to_hex(), "frameSize": 0, "stride": 0,
                        "leftContext": 0, "rightContext": 0, "nostrEvent": event.as_json(),
                        "sessions": event.id().to_hex(), "isBot": str(is_bot), "startTime": "0", "endTime": "0"}

        if task == "speech-to-text":
            # Declare specific model type e.g. whisperx_large-v2
            request_form["mode"] = "PREDICT"
            alignment = "raw"
            model_option = "large-v2"

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

                    elif param == "alignment":
                        alignment = tag.as_vec()[2]
                    elif param == "model":
                        model_option = tag.as_vec()[2]

            request_form["schemeType"] = "FREE"
            request_form["scheme"] = "transcript"
            request_form["streamName"] = "audio"
            request_form["trainerFilePath"] = 'models\\trainer\\' + str(
                request_form["schemeType"]).lower() + '\\' + str(
                request_form["scheme"]) + '\\audio{audio}\\whisperx\\whisperx_transcript.trainer'
            request_form["optStr"] = 'model=' + model_option + ';alignment_mode=' + alignment + ';batch_size=2'

        elif task == "translation":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'translation'
            input_type = "event"
            text = ""
            translation_lang = "en"
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    input_type = tag.as_vec()[2]

                elif tag.as_vec()[0] == 'param':
                    param = tag.as_vec()[1]
                    if param == "language":  # check for paramtype
                        translation_lang = str(tag.as_vec()[2]).split('-')[0]

            if input_type == "event":
                for tag in event.tags():
                    if tag.as_vec()[0] == 'i':
                        evt = get_event_by_id(tag.as_vec()[1])
                        text = evt.content()
                        break

            elif input_type == "text":
                for tag in event.tags():
                    if tag.as_vec()[0] == 'i':
                        text = tag.as_vec()[1]
                        break

            elif input_type == "job":
                for tag in event.tags():
                    if tag.as_vec()[0] == 'i':
                        job_id_filter = Filter().kind(65001).event(EventId.from_hex(tag.as_vec()[1])).limit(1)
                        events = client.get_events_of([job_id_filter], timedelta(seconds=DVMConfig.RELAY_TIMEOUT))
                        evt = events[0]
                        text = evt.content()
                        break

            request_form["optStr"] = 'text=' + text + ';translation_lang=' + translation_lang

        elif task == "image-to-text":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'image-to-text'
            input_type = "url"
            url = ""
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    if input_type == "url":
                        url = tag.as_vec()[1]
                    elif input_type == "event":
                        evt = get_event_by_id(tag.as_vec()[1])
                        url = re.search("(?P<url>https?://[^\s]+)", evt.content()).group("url")
                    elif input_type == "job":
                        job_id_filter = Filter().kind(65001).event(EventId.from_hex(tag.as_vec()[1])).limit(1)
                        events = client.get_events_of([job_id_filter], timedelta(seconds=DVMConfig.RELAY_TIMEOUT))
                        evt = events[0]
                        url = re.search("(?P<url>https?://[^\s]+)", evt.content()).group("url")
            request_form["optStr"] = 'url=' + url

        elif task == "image-to-image":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'image-to-image'
            prompt = ""
            url = ""
            negative_prompt = ""
            strength = 0.5
            guidance_scale = 7.5
            model = "sdxl"

            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    input_type = tag.as_vec()[2]
                    if input_type == "url":
                        url = tag.as_vec()[1]
                    elif input_type == "event":
                        evt = get_event_by_id(tag.as_vec()[1])
                        url = re.search("(?P<url>https?://[^\s]+)", evt.content()).group("url")
                    elif input_type == "job":
                        job_id_filter = Filter().kind(65001).event(EventId.from_hex(tag.as_vec()[1])).limit(1)
                        events = client.get_events_of([job_id_filter], timedelta(seconds=DVMConfig.RELAY_TIMEOUT))
                        evt = events[0]
                        url = evt.content()
                elif tag.as_vec()[0] == 'param':
                    if tag.as_vec()[1] == "prompt":
                        prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "negative_prompt":
                        negative_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "strength":
                        strength = float(tag.as_vec()[2])
                    elif tag.as_vec()[1] == "guidance_scale":
                        guidance_scale = float(tag.as_vec()[2])
                    elif tag.as_vec()[1] == "model":
                        model = tag.as_vec()[2]

            request_form["optStr"] = ('url=' + url + ';prompt=' + prompt + ';negative_prompt=' + negative_prompt
                                     + ';strength=' + str(strength) + ';guidance_scale=' + str(guidance_scale)
                                     + ';model=' + model)

        elif task == "text-to-image":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'text-to-image'
            prompt = ""
            extra_prompt = ""
            negative_prompt = ""
            upscale = "4"
            model = "stabilityai/stable-diffusion-xl-base-1.0"

            width = "1024"
            height = "1024"
            ratio_width = "1"
            ratio_height = "1"

            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    input_type = tag.as_vec()[2]
                    if input_type == "text":
                        prompt = tag.as_vec()[1]
                    elif input_type == "event":
                        evt = get_event_by_id(tag.as_vec()[1])
                        prompt = evt.content()
                    elif input_type == "job":
                        job_id_filter = Filter().kind(65001).event(EventId.from_hex(tag.as_vec()[1])).limit(1)
                        events = client.get_events_of([job_id_filter], timedelta(seconds=DVMConfig.RELAY_TIMEOUT))
                        evt = events[0]
                        prompt = evt.content()
                elif tag.as_vec()[0] == 'param':
                    if tag.as_vec()[1] == "prompt":
                        extra_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "negative_prompt":
                        negative_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "size":
                        width = tag.as_vec()[2]
                        height = tag.as_vec()[3]
                    elif tag.as_vec()[1] == "ratio":
                        ratio_width = (tag.as_vec()[2])
                        ratio_height = (tag.as_vec()[3])
                    elif tag.as_vec()[1] == "upscale":
                        upscale = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "model":
                        model = tag.as_vec()[2]

            request_form["optStr"] = ('prompt=' + prompt + ';extra_prompt=' + extra_prompt + ';negative_prompt='
                                      + negative_prompt + ';width=' + str(width) + ';height=' + str(height)
                                      + ';upscale=' + str(upscale) + ';model=' + model + ';ratiow=' + str(ratio_width)
                                      + ';ratioh=' + str(ratio_height))

        elif task == "image-upscale":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'image-upscale'
            upscale = "4"
            url = ""
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    input_type = tag.as_vec()[2]
                    if input_type == "url":
                        url = tag.as_vec()[1]
                        print(url)
                    elif input_type == "event":
                        evt = get_event_by_id(tag.as_vec()[1])
                        url = re.search("(?P<url>https?://[^\s]+)", evt.content()).group("url")
                    elif input_type == "job":
                        job_id_filter = Filter().kind(65001).event(EventId.from_hex(tag.as_vec()[1])).limit(1)
                        events = client.get_events_of([job_id_filter], timedelta(seconds=DVMConfig.RELAY_TIMEOUT))
                        evt = events[0]
                        url = evt.content()
                elif tag.as_vec()[0] == 'param':
                    if tag.as_vec()[1] == "upscale":
                        upscale = tag.as_vec()[2]

            request_form["optStr"] = 'url=' + url + ";upscale=" + upscale

        elif task == "chat":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'chat'
            text = ""
            user = event.pubkey().to_hex()
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    text = tag.as_vec()[1].replace(";", "")
                elif tag.as_vec()[0] == 'p':
                    user = tag.as_vec()[1]
            request_form["optStr"] = 'message=' + text + ';user=' + user


        elif task == "summarization":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'summarization'
            text = ""
            user = event.pubkey().to_hex()
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    input_type = tag.as_vec()[2]
                    if input_type == "text":
                        text = tag.as_vec()[1].replace(";", "")
                    elif input_type == "event":
                        evt = get_event_by_id(tag.as_vec()[1])
                        text = evt.content().replace(";", "")
                    elif input_type == "job":
                        job_id_filter = Filter().kind(65001).event(EventId.from_hex(tag.as_vec()[1])).limit(1)
                        events = client.get_events_of([job_id_filter], timedelta(seconds=DVMConfig.RELAY_TIMEOUT))
                        if len(events) > 0:
                            text = events[0].content().replace(";", "")
                elif tag.as_vec()[0] == 'p':
                    user = tag.as_vec()[1]
                request_form["optStr"] = 'message=' + text + ';user=' + user

        elif task == "inactive-following":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'inactive-following'
            days = "30"
            number = "25"
            user = event.pubkey().to_hex()
            for tag in event.tags():
                if tag.as_vec()[0] == 'p':
                    user = tag.as_vec()[1]
                elif tag.as_vec()[0] == 'param':
                    if tag.as_vec()[1] == 'since':
                        days = tag.as_vec()[2]
                    elif tag.as_vec()[1] == 'numusers':
                        number = tag.as_vec()[2]
            request_form["optStr"] = 'user=' + user + ';since=' + days + ';num=' + number

        return request_form

    def organize_input_data(event, request_form):
        data_dir = os.environ["NOVA_DATA_DIR"]

        session = event.id().to_hex()
        input_type = "url"
        input_value = ""
        for tag in event.tags():
            if tag.as_vec()[0] == 'i':
                input_value = tag.as_vec()[1]
                input_type = tag.as_vec()[2]
                break

        if input_type == "url":
            if not os.path.exists(data_dir + '\\' + request_form["database"] + '\\' + session):
                os.mkdir(data_dir + '\\' + request_form["database"] + '\\' + session)
            # We can support some services that don't use default media links, like overcastfm for podcasts
            if str(input_value).startswith("https://overcast.fm/"):
                filename = data_dir + '\\' + request_form["database"] + '\\' + session + '\\' + request_form[
                    "roles"] + ".originalaudio.mp3"
                print("Found overcast.fm Link.. downloading")
                download_podcast(input_value, filename)
                finaltag = str(input_value).replace("https://overcast.fm/", "").split('/')
                if float(request_form["startTime"]) == 0.0:
                    if len(finaltag) > 1:
                        t = time.strptime(finaltag[1], "%H:%M:%S")
                        seconds = t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec
                        request_form["startTime"] = str(seconds)  # overwrite from link.. why not..
                        print("Setting start time automatically to " + request_form["startTime"])
                        if float(request_form["endTime"]) > 0.0:
                            request_form["endTime"] = seconds + float(request_form["endTime"])
                            print("Moving end time automatically to " + request_form["endTime"])

            # or youtube links..
            elif str(input_value).replace("http://", "").replace("https://", "").replace(
                    "www.", "").replace("youtu.be/", "youtube.com?v=")[0:11] == "youtube.com":
                filepath = data_dir + '\\' + request_form["database"] + '\\' + session + '\\'
                try:
                    filename = downloadYouTube(input_value, filepath)
                    o = urlparse(input_value)
                    q = urllib.parse.parse_qs(o.query)
                except Exception as e:
                    print(e)
                    return None

                if float(request_form["startTime"]) == 0.0:
                    if o.query.find('t=') != -1:
                        request_form["startTime"] = q['t'][0]  # overwrite from link.. why not..
                        print("Setting start time automatically to " + request_form["startTime"])
                        if float(request_form["endTime"]) > 0.0:
                            request_form["endTime"] = str(float(q['t'][0]) + float(request_form["endTime"]))
                            print("Moving end time automatically to " + request_form["endTime"])

            # Regular links have a media file ending and/or mime types
            else:
                req = requests.get(input_value)
                content_type = req.headers['content-type']
                print(content_type)
                if content_type == 'audio/x-wav' or str(input_value).lower().endswith(".wav"):
                    ext = "wav"
                    file_type = "audio"
                elif content_type == 'audio/mpeg' or str(input_value).lower().endswith(".mp3"):
                    ext = "mp3"
                    file_type = "audio"
                elif content_type == 'audio/ogg' or str(input_value).lower().endswith(".ogg"):
                    ext = "ogg"
                    file_type = "audio"
                elif content_type == 'video/mp4' or str(input_value).lower().endswith(".mp4"):
                    ext = "mp4"
                    file_type = "video"
                elif content_type == 'video/avi' or str(input_value).lower().endswith(".avi"):
                    ext = "avi"
                    file_type = "video"
                elif content_type == 'video/quicktime' or str(input_value).lower().endswith(".mov"):
                    ext = "mov"
                    file_type = "video"

                else:
                    print(str(input_value).lower())
                    return None

                filename = data_dir + '\\' + request_form["database"] + '\\' + session + '\\' + request_form[
                    "roles"] + '.original' + file_type + '.' + ext
                print(filename)

                try:
                    if not os.path.exists(filename):
                        file = open(filename, 'wb')
                        for chunk in req.iter_content(100000):
                            file.write(chunk)
                        file.close()
                except Exception as e:
                    print(e)
            try:
                file_reader = AudioReader(filename, ctx=cpu(0), mono=False)
                duration = file_reader.duration()
            except Exception as e:
                print(e)
                return None

            print("Duration of the Media file: " + str(duration))
            if float(request_form['endTime']) == 0.0:
                end_time = float(duration)
            elif float(request_form['endTime']) > duration:
                end_time = float(duration)
            else:
                end_time = float(request_form['endTime'])
            if float(request_form['startTime']) < 0.0 or float(request_form['startTime']) > end_time:
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
        return True

    def do_work(job_event, is_from_bot=False):
        if (65002 <= job_event.kind() <= 66000) or job_event.kind() == 4 or job_event.kind() == 68001:
            request_form = create_requestform_from_nostr_event(job_event, is_from_bot)
            task = get_task(job_event)
            if task == "speech-to-text":
                print("[Nostr] Adding Nostr speech-to-text Job event: " + job_event.as_json())
                if organize_input_data(job_event, request_form) is None:
                    respond_to_error("Error processing video", job_event.as_json(), is_from_bot)
                    return
            elif task == "event-list-generation" or task.startswith("unknown"):
                print("Task not (yet) supported")
                return
            else:
                print("[Nostr] Adding " + task + " Job event: " + job_event.as_json())

            url = 'http://' + os.environ["NOVA_HOST"] + ':' + os.environ["NOVA_PORT"] + '/' + str(
                request_form["mode"]).lower()
            headers = {'Content-type': 'application/x-www-form-urlencoded'}
            requests.post(url, headers=headers, data=request_form)

    client.handle_notifications(NotificationHandler())

    while True:
        for job in job_list:
            if job.bolt11 != "" and job.payment_hash != "" and not job.is_paid:
                if str(check_bolt11_ln_bits_is_paid(job.payment_hash)) == "True":
                    job.is_paid = True
                    event = get_event_by_id(job.event_id)
                    send_job_status_reaction(event, "processing", True, 0, client=client)
                    do_work(event, is_from_bot=False)

            if Timestamp.now().as_secs() > job.expires:
                job_list.remove(job)

        #if len(job_list) > 0:
        #    print(str(job_list))

        time.sleep(5.0)


# SEND AND RECEIVE EVENTS
def get_event_by_id(event_id_hex, client=None):
    is_new_client = False
    if client is None:
        keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
        client = Client(keys)
        for relay in DVMConfig.RELAY_LIST:
            client.add_relay(relay)
        client.connect()
        is_new_client = True

    id_filter = Filter().id(event_id_hex).limit(1)
    events = client.get_events_of([id_filter], timedelta(seconds=DVMConfig.RELAY_TIMEOUT))
    if is_new_client:
        client.disconnect()
    if len(events) > 0:
        return events[0]
    else:
        return None


def send_event(event, client=None):
    relays = []
    is_new_client = False

    for tag in event.tags():
        if tag.as_vec()[0] == 'relays':
            relays = tag.as_vec()[1].split(',')

    if client is None:
        keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
        client = Client(keys)
        for relay in DVMConfig.RELAY_LIST:
            client.add_relay(relay)
        client.connect()
        is_new_client = True

    for relay in relays:
        if relay not in DVMConfig.RELAY_LIST:
            client.add_relay(relay)
    client.connect()

    event_id = client.send_event(event)

    for relay in relays:
        if relay not in DVMConfig.RELAY_LIST:
            client.remove_relay(relay)

    if is_new_client:
        client.disconnect()

    return event_id


# GET INFO ON TASK
def get_task(event):
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
                    file_type = check_url_is_readable(tag.as_vec()[1])
                    if file_type == "audio" or file_type == "video":
                        return "speech-to-text"
                    elif file_type == "image":
                        return "image-to-text"
                    else:
                        return "unknown job"
                else:
                    return "unknown type"
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


def get_amount_per_task(task):
    if task == "translation":
        duration = 1  # todo get task duration
        amount = DVMConfig.COSTPERUNIT_TRANSLATION * duration
    elif task == "speech-to-text":
        duration = 1  # todo get task duration
        amount = DVMConfig.COSTPERUNIT_SPEECHTOTEXT * duration
    elif task == "text-to-image":
        amount = DVMConfig.COSTPERUNIT_IMAGEGENERATION
    elif task == "image-to-image":
        amount = DVMConfig.COSTPERUNIT_IMAGETRANSFORMING
    elif task == "image-upscale":
        amount = DVMConfig.COSTPERUNIT_IMAGEUPSCALING
    elif task == "chat":
        amount = 0
    elif task == "image-to-text":
        amount = DVMConfig.COSTPERUNIT_IMAGEGENERATION
    elif task == "summarization":
        amount = DVMConfig.COSTPERUNIT_IMAGEGENERATION
    elif task == "inactive-following":
        amount = DVMConfig.COSTPERUNIT_INACTIVE_FOLLOWING
    else:
        print("[Nostr] Task " + task + " is currently not supported by this instance, skipping")
        return None
    return amount


# DECIDE TO RETURN RESULT
def check_event_status(data, original_event_str: str, use_bot=False):
    original_event = Event.from_json(original_event_str)
    for x in job_list:
        if x.event_id == original_event.id().to_hex():
            is_paid = x.is_paid
            amount = x.amount
            x.result = data
            x.is_processed = True
            if DVMConfig.SHOWRESULTBEFOREPAYMENT and not is_paid:
                send_nostr_reply_event(data, original_event_str)
                send_job_status_reaction(original_event, "success", amount)  # or payment-required, or both?
            elif not DVMConfig.SHOWRESULTBEFOREPAYMENT and not is_paid:
                send_job_status_reaction(original_event, "success", amount)  # or payment-required, or both?

            if DVMConfig.SHOWRESULTBEFOREPAYMENT and is_paid:
                job_list.remove(x)
            elif not DVMConfig.SHOWRESULTBEFOREPAYMENT and is_paid:
                job_list.remove(x)
                send_nostr_reply_event(data, original_event_str)
            break

    post_processed_content = post_process_result(data, original_event)
    print(str(job_list))
    if use_bot:
        keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
        receiver_key = PublicKey()
        for tag in original_event.tags():
            if tag.as_vec()[0] == "p":
                receiver_key = PublicKey.from_hex(tag.as_vec()[1])
        event = EventBuilder.new_encrypted_direct_msg(keys, receiver_key, post_processed_content, None).to_event(keys)
        send_event(event)

    else:
        send_nostr_reply_event(post_processed_content, original_event_str)
        send_job_status_reaction(original_event, "success")


# NIP90 REPLIES
def respond_to_error(content, originaleventstr, is_from_bot=False):
    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    original_event = Event.from_json(originaleventstr)
    sender = ""
    task = ""
    if not is_from_bot:
        send_job_status_reaction(original_event, "error", content=content)
        # TODO Send Zap back
    else:
        for tag in original_event.tags():
            if tag.as_vec()[0] == "p":
                sender = tag.as_vec()[1]
            elif tag.as_vec()[0] == "i":
                task = tag.as_vec()[1]

        user = get_from_sql_table(sender)
        is_whitelisted = user[2]
        if not is_whitelisted:
            update_sql_table(sender, user[1] + get_amount_per_task(task), user[2], user[3], user[4], user[5], user[6],
                             Timestamp.now().as_secs())
            message = "There was the following error : " + content + ". Credits have been reimbursed"
        else:
            # User didn't pay, so no reimbursement
            message = "There was the following error : " + content

        evt = EventBuilder.new_encrypted_direct_msg(keys, PublicKey.from_hex(sender), message, None).to_event(keys)
        send_event(evt)


def send_nostr_reply_event(content, original_event_as_str):
    originalevent = Event.from_json(original_event_as_str)
    requesttag = Tag.parse(["request", original_event_as_str.replace("\\", "")])
    etag = Tag.parse(["e", originalevent.id().to_hex()])
    ptag = Tag.parse(["p", originalevent.pubkey().to_hex()])
    alttag = Tag.parse(["alt", "This is the result of a NIP90 DVM AI task with kind " + str(
        originalevent.kind()) + ". The task was: " + originalevent.content()])
    statustag = Tag.parse(["status", "success"])

    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    event = EventBuilder(65001, str(content), [requesttag, etag, ptag, alttag, statustag]).to_event(keys)
    send_event(event)
    print("[Nostr] 65001 Job Response event sent: " + event.as_json())
    return event.as_json()


def send_job_status_reaction(original_event, status, is_paid=True, amount=0, client=None, content=None):
    altdesc = "This is a reaction to a NIP90 DVM AI task. "
    task = get_task(original_event)
    if status == "processing":
        altdesc = "NIP90 DVM AI task " + task + " started processing. "
        reaction = altdesc + emoji.emojize(":thumbs_up:")
    elif status == "success":
        altdesc = "NIP90 DVM AI task " + task + " finished successfully. "
        reaction = altdesc + emoji.emojize(":call_me_hand:")
    elif status == "error":
        altdesc = "NIP90 DVM AI task " + task + " had an error. "
        if content is None:
            reaction = altdesc + emoji.emojize(":thumbs_down:")
        else:
            reaction = altdesc + emoji.emojize(":thumbs_down:") + content

    elif status == "payment-required":

        altdesc = "NIP90 DVM AI task " + task + " requires payment of min " + str(amount) + " Sats. "
        if task == "speech-to-text":
            altdesc = altdesc + (" Providing results with WhisperX large-v2. "
                                 "Accepted input formats: wav,mp3,mp4,ogg,avi,mov,youtube,overcast. "
                                 "Possible outputs: text/plain, timestamped labels depending on "
                                 "alignment parameter (word,segment,raw) ")
        elif task == "image-to-text":
            altdesc = altdesc + (" Accepted input formats: jpg. Possible outputs: text/plain. "
                                 "This is very experimental, make sure your text is well readable. ")
        reaction = altdesc + emoji.emojize(":orange_heart:")

    elif status == "payment-rejected":
        altdesc = "NIP90 DVM AI task " + task + " payment is below required amount of " + str(amount) + " Sats. "
        reaction = altdesc + emoji.emojize(":see_no_evil_monkey:")
    elif status == "user-blocked-from-service":

        altdesc = "NIP90 DVM AI task " + task + " can't be performed. User has been blocked from Service. "
        reaction = altdesc + emoji.emojize(":see_no_evil_monkey:")
    else:
        reaction = emoji.emojize(":see_no_evil_monkey:")

    etag = Tag.parse(["e", original_event.id().to_hex()])
    ptag = Tag.parse(["p", original_event.pubkey().to_hex()])
    alttag = Tag.parse(["alt", altdesc])
    statustag = Tag.parse(["status", status])
    tags = [etag, ptag, alttag, statustag]

    if status == "success" or status == "error":  #
        for x in job_list:
            if x.event_id == original_event.id():
                is_paid = x.is_paid
                amount = x.amount
                break

    bolt11 = ""
    expires = original_event.created_at().as_secs() + (60*60*24)
    if status == "payment-required" or (status == "processing" and not is_paid):
        if DVMConfig.LNBITS_INVOICE_KEY != "":
            try:
                bolt11, payment_hash = create_bolt11_ln_bits(amount)
            except Exception as e:
                print(e)
        job_list.append(
            JobToWatch(event_id=original_event.id().to_hex(), timestamp=original_event.created_at().as_secs(), amount=amount,
                       is_paid=is_paid,
                       status=status, result="", is_processed=False, bolt11=bolt11, payment_hash=payment_hash, expires=expires, from_bot=False))
        print(str(job_list))
    if status == "payment-required" or status == "payment-rejected" or (status == "processing" and not is_paid) or (
            status == "success" and not is_paid):

        if DVMConfig.LNBITS_INVOICE_KEY != "":
            amount_tag = Tag.parse(["amount", str(amount * 1000), bolt11])
        else:
            amount_tag = Tag.parse(["amount", str(amount * 1000)])  # to millisats
        tags.append(amount_tag)




    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    event = EventBuilder(65000, reaction, tags).to_event(keys)

    send_event(event, client)
    print("[Nostr] Sent Kind 65000 Reaction: " + status + " " + event.as_json())
    return event.as_json()


# POSTPROCESSING
def post_process_result(anno, original_event):
    print("post-processing...")
    if isinstance(anno, Anno): #if input is an anno we parse it to required output format
        for tag in original_event.tags():
            if tag.as_vec()[0] == "output":
                print("requested output is " +tag.as_vec()[1] + "...")
                try:
                    if tag.as_vec()[1] == "text/plain":
                        result = ""
                        for element in anno.data:
                            name = element["name"] #name
                            cleared_name = str(name).lstrip("\'").rstrip("\'")
                            result = result + cleared_name + "\n"
                        result = str(result).replace("\"", "").replace('[', "").replace(']', "").lstrip(None)

                    elif tag.as_vec()[1] == "text/json" or tag.as_vec()[1] == "json":
                        #result = json.dumps(json.loads(anno.data.to_json(orient="records")))
                        result = json.dumps(anno.data.tolist())
                    # TODO add more
                    else:
                        result = str(anno.data)
                except Exception as e:
                    print(e)
                    result = str(anno.data)

    elif isinstance(anno, str): #If input is a string we do nothing for now.
        result = anno
    return result


# BOT FUNCTIONS
def get_bot_help_text():
    return (
            "Hi there. I'm a bot interface to the first NIP90 Data Vending Machine and I can perform several AI tasks"
            " for you. Currently I can do the following jobs:\n\n"
            "Generate an Image with Stable Diffusion XL (" + str(DVMConfig.COSTPERUNIT_IMAGEGENERATION) + " Sats)\n"
            "-text-to-image someprompt\nAdditional parameters:\n-negative some negative prompt\n-ratio width:height "
            "(e.g. 3:4), default 1:1\n-model anothermodel\nOther Models are: realistic, wild, sd15, lora_ghibli, "
            "lora_monster, lora_inks, lora_t4, lora_pokemon\n\n"
            "Transform an existing Image with Stable Diffusion XL (" + str(DVMConfig.COSTPERUNIT_IMAGETRANSFORMING)
            + " Sats)\n" "-image-to-image urltoimage -prompt someprompt\n\n"
            "Parse text from an Image (make sure text is well readable) (" + str(DVMConfig.COSTPERUNIT_OCR) + " Sats)\n"
            "-image-to-text urltofile \n\n"
            "Upscale the resolution of an Image 4x and improve quality (" + str(DVMConfig.COSTPERUNIT_IMAGEUPSCALING)
            + " Sats)\n -image-upscale urltofile \n\n"
            "Transcribe Audio/Video/Youtube/Overcast from an URL with WhisperX large-v2 (" + str(
            DVMConfig.COSTPERUNIT_SPEECHTOTEXT) + " Sats)\n"
            "-speech-to-text urltofile \nAdditional parameters:\n-from timeinseconds -to timeinseconds\n\n"
            "Get a List of 25 inactive users you follow (" + str(DVMConfig.COSTPERUNIT_INACTIVE_FOLLOWING) + " Sats)\n"
            "-inactive-following\nAdditional parameters:\n-sincedays days (e.g. 60), default 30\n\n"
            "To show your current balance\n -balance \n\n"
            "You can either zap my responses directly if your client supports it (e.g. Amethyst) or you can zap any "
            "post or my profile (e.g. in Damus) to top up your balance.")


def parse_bot_command_to_event(dec_text):
    dec_text = dec_text.replace("\n", "")
    if str(dec_text).startswith("-text-to-image"):
        command = dec_text.replace("-text-to-image ", "")
        split = command.split(" -")
        prompt = split[0]
        width = "1024"
        height = "1024"
        ratiow = "1"
        ratioh = "1"
        j_tag = Tag.parse(["j", "text-to-image"])
        i_tag = Tag.parse(["i", prompt, "text"])
        tags = [j_tag, i_tag]
        if len(split) > 1:
            for i in split:
                if i.startswith("negative "):
                    negative_prompt = i.replace("negative ", "")
                    param_tag = Tag.parse(["param", "negative_prompt", negative_prompt])
                    tags.append(param_tag)
                elif i.startswith("extra "):
                    extra_prompt = i.replace("extra ", "")
                    param_tag = Tag.parse(["param", "prompt", extra_prompt])
                    tags.append(param_tag)
                elif i.startswith("upscale "):
                    upscale_factor = i.replace("upscale ", "")
                    param_tag = Tag.parse(["param", "upscale", upscale_factor])
                    tags.append(param_tag)
                elif i.startswith("model "):
                    model = i.replace("model ", "")
                    param_tag = Tag.parse(["param", "model", model])
                    tags.append(param_tag)
                elif i.startswith("width "):
                    width = i.replace("width ", "")
                elif i.startswith("height "):
                    height = i.replace("height ", "")
                elif i.startswith("ratio "):
                    ratio = str(i.replace("ratio ", ""))
                    split = ratio.split(":")
                    ratiow = split[0]
                    ratioh = split[1]

        param_ratio_tag = Tag.parse(["param", "ratio", ratiow, ratioh])
        param_size_tag = Tag.parse(["param", "size", width, height])
        tags.append(param_ratio_tag)
        tags.append(param_size_tag)

        return tags

    elif str(dec_text).startswith("-image-to-image"):
        command = dec_text.replace("-image-to-image ", "")
        split = command.split(" -")
        url = str(split[0]).replace(' ', '')
        width = "768"
        height = "768"
        j_tag = Tag.parse(["j", "image-to-image"])
        i_tag = Tag.parse(["i", url, "url"])
        tags = [j_tag, i_tag]
        if len(split) > 1:
            for i in split:
                if i.startswith("negative "):
                    negative_prompt = i.replace("negative ", "")
                    param_tag = Tag.parse(["param", "negative_prompt", negative_prompt])
                    tags.append(param_tag)
                elif i.startswith("prompt "):
                    prompt = i.replace("prompt ", "")
                    param_tag = Tag.parse(["param", "prompt", prompt])
                    tags.append(param_tag)
                elif i.startswith("strength "):
                    strength = i.replace("strength ", "")
                    param_tag = Tag.parse(["param", "strength", strength])
                    tags.append(param_tag)
                elif i.startswith("guidance_scale "):
                    strength = i.replace("guidance_scale ", "")
                    param_tag = Tag.parse(["param", "guidance_scale", strength])
                    tags.append(param_tag)
                elif i.startswith("model "):
                    model = i.replace("model ", "")
                    param_tag = Tag.parse(["param", "model", model])
                    tags.append(param_tag)

            param_size_tag = Tag.parse(["param", "size", width, height])
            tags.append(param_size_tag)

            return tags

    elif str(dec_text).startswith("-image-upscale"):
        command = dec_text.replace("-image-upscale ", "")
        split = command.split(" -")
        url = str(split[0]).replace(' ', '')
        j_tag = Tag.parse(["j", "image-upscale"])
        i_tag = Tag.parse(["i", url, "url"])
        tags = [j_tag, i_tag]
        if len(split) > 1:
            for i in split:
                if i.startswith("upscale "):
                    upscale_factor = i.replace("upscale ", "")
                    param_tag = Tag.parse(["param", "upscale", upscale_factor])
                    tags.append(param_tag)
        return tags

    elif str(dec_text).startswith("-image-to-text"):
        command = dec_text.replace("-image-to-text ", "")
        split = command.split(" -")
        url = str(split[0]).replace(' ', '')
        j_tag = Tag.parse(["j", "image-to-text"])
        i_tag = Tag.parse(["i", url, "url"])
        tags = [j_tag, i_tag]
        return tags

    elif str(dec_text).startswith("-speech-to-text"):
        command = dec_text.replace("-speech-to-text ", "")
        split = command.split(" -")
        url = split[0]
        start = "0"
        end = "0"
        model = "large-v2"
        if len(split) > 1:
            for i in split:
                if i.startswith("from "):
                    start = i.replace("from ", "")
                elif i.startswith("to "):
                    end = i.replace("to ", "")
                elif i.startswith("model "):
                    model = i.replace("model ", "")
        j_tag = Tag.parse(["j", "speech-to-text"])
        i_tag = Tag.parse(["i", url, "url"])
        o_tag = Tag.parse(["output", "text/plain"])
        param_tag_since = Tag.parse(["param", "model", model])
        param_tag = Tag.parse(["param", "range", start, end])
        return [j_tag, i_tag, o_tag, param_tag_since, param_tag]

    elif str(dec_text).startswith("-inactive-following"):
        since_days = "30"
        number_users = "25"
        command = dec_text.replace("-inactive-following ", "")
        split = command.split(" -")
        for i in split:
            if i.startswith("sincedays "):
                since_days = i.replace("sincedays ", "")
            elif i.startswith("num"):
                number_users = i.replace("num ", "")
        param_tag_since = Tag.parse(["param", "since", since_days])
        param_tag_num_users = Tag.parse(["param", "numusers", number_users])
        j_tag = Tag.parse(["j", "inactive-following"])
        return [j_tag, param_tag_since, param_tag_num_users]
    else:
        text = dec_text
        j_tag = Tag.parse(["j", "chat"])
        i_tag = Tag.parse(["i", text, "text"])
        return [j_tag, i_tag]


# CHECK INPUTS/TASK AVAILABLE
def check_task_is_supported(event):

    input_value = ""
    input_type = ""
    for tag in event.tags():
        if tag.as_vec()[0] == 'i':
            if len(tag.as_vec()) < 3:
                print("Job Event missing/malformed i tag, skipping..")
                return False
            else:
                input_value = tag.as_vec()[1]
                input_type = tag.as_vec()[2]
        elif tag.as_vec()[0] == 'output':
            output = tag.as_vec()[1]
            if not (output == "text/plain" or output == "text/json" or output == "json"):
                print("Output format not supported, skipping..")
                return False

    task = get_task(event)
    print("Received new Task: " + task)

    if task not in DVMConfig.SUPPORTED_TASKS:  # The Tasks this DVM supports (can be extended)
        print("Task not supported, skipping..")
        return False
    if task == "translation" and (
            input_type != "event" and input_type != "job" and input_type != "text"):  # The input types per task
        return False
    # if task == "translation" and len(event.content) > 4999:  # Google Services have a limit of 5000 signs
    #    return False
    if task == "speech-to-text" and (
            input_type != "event" and input_type != "job" and input_type != "url"):  # The input types per task
        return False
    if task == "image-upscale" and (input_type != "event" and input_type != "job" and input_type != "url"):
        return False
    if input_type == 'url' and check_url_is_readable(input_value) is None:
        return False

    return True


def check_url_is_readable(url):
    if not str(url).startswith("http"):
        return None
    # If it's a YouTube oder Overcast link, we suppose we support it
    if str(url).replace("http://", "").replace("https://", "").replace("www.", "").replace("youtu.be/",
                                                                                           "youtube.com?v=")[
       0:11] == "youtube.com" and str(url).find("live") == -1:
        # print("CHECKING YOUTUBE")f
        # if (checkYoutubeLinkValid(url)):
        return "video"

    elif str(url).startswith("https://overcast.fm/"):
        return "audio"

    # If link is comaptible with one of these file formats, it's fine.
    req = requests.get(url)
    content_type = req.headers['content-type']
    if content_type == 'audio/x-wav' or str(url).endswith(".wav") or content_type == 'audio/mpeg' or str(url).endswith(
            ".mp3") or content_type == 'audio/ogg' or str(url).endswith(".ogg"):
        return "audio"
    elif content_type == 'image/png' or str(url).endswith(".png") or content_type == 'image/jpg' or str(url).endswith(
            ".jpg") or content_type == 'image/jpeg' or str(url).endswith(".jpeg") or content_type == 'image/png' or str(
            url).endswith(".png"):
        return "image"
    elif content_type == 'video/mp4' or str(url).endswith(".mp4") or content_type == 'video/avi' or str(url).endswith(
            ".avi") or content_type == 'video/mov' or str(url).endswith(".mov"):
        return "video"
    # Otherwise we will not offer to do the job.
    return None


# NOVADB CONNECTION CONFIG
def save_config(db_user, db_password, db_server, database, role, annotator):
    # Get the configparser object
    config_object = ConfigParser()

    # Assume we need 2 sections in the config file, let's call them USERINFO and SERVERCONFIG
    config_object["USERINFO"] = {
        "dbUser": db_user,
        "dbPassword": db_password
    }

    config_object["SERVERCONFIG"] = {
        "dbServer": db_server,
        "database": database,
        "roles": role,
        "annotator": annotator
    }

    # Write the above sections to config.ini file
    with open('nostrconfig.ini', 'w') as conf:
        config_object.write(conf)


# LIGHTNING FUNCTIONS
def parse_bolt11_invoice(invoice):
    remaining_invoice = invoice[4:]
    index = get_index_of_first_letter(remaining_invoice)
    identifier = remaining_invoice[index]
    number_string = remaining_invoice[:index]
    number = float(number_string)
    if identifier == 'm':
        number = number * 100000000 * 0.001
    elif identifier == 'u':
        number = number * 100000000 * 0.000001
    elif identifier == 'n':
        number = number * 100000000 * 0.000000001
    elif identifier == 'p':
        number = number * 100000000 * 0.000000000001

    return int(number)


def create_bolt11_ln_bits(sats):
    url = DVMConfig.LNBITS_URL + "/api/v1/payments"
    data = {'out': False, 'amount': sats, 'memo': "Nostr-DVM"}
    headers = {'X-API-Key': DVMConfig.LNBITS_INVOICE_KEY, 'Content-Type': 'application/json', 'charset': 'UTF-8'}
    try:
        res = requests.post(url, json=data, headers=headers)
        obj = json.loads(res.text)
        return obj["payment_request"], obj["payment_hash"]
    except Exception as e:
        print(e)
        return None

def check_bolt11_ln_bits_is_paid(payment_hash):
    url = DVMConfig.LNBITS_URL + "/api/v1/payments/" + payment_hash
    headers = {'X-API-Key': DVMConfig.LNBITS_INVOICE_KEY, 'Content-Type': 'application/json', 'charset': 'UTF-8'}
    try:
        res = requests.get(url, headers=headers)
        obj = json.loads(res.text)
        return obj["paid"]
    except Exception as e:
        print(e)
        return None


def get_index_of_first_letter(ip):
    index = 0
    for c in ip:
        if c.isalpha():
            return index
        else:
            index = index + 1
    return len(ip)


# DECRYPT ZAPS
def check_for_zapplepay(sender, content):
    try:
        # Special case Zapplepay
        if sender == PublicKey.from_bech32("npub1wxl6njlcgygduct7jkgzrvyvd9fylj4pqvll6p32h59wyetm5fxqjchcan").to_hex():
            real_sender_bech32 = content.replace("From: nostr:", "")
            sender = PublicKey.from_bech32(real_sender_bech32).to_hex()
        return sender

    except Exception as e:
        print(e)
        return sender


def decrypt_private_zap_message(msg, privkey, pubkey):
    shared_secret = nostr_sdk.generate_shared_key(privkey, pubkey)
    if len(shared_secret) != 16 and len(shared_secret) != 32:
        return "invalid shared secret size"
    parts = msg.split("_")
    if len(parts) != 2:
        return "invalid message format"
    try:
        _, encrypted_msg = bech32_decode(parts[0])
        encrypted_bytes = convertbits(encrypted_msg, 5, 8, False)
        _, iv = bech32_decode(parts[1])
        iv_bytes = convertbits(iv, 5, 8, False)
    except Exception as e:
        return e
    try:
        cipher = AES.new(bytearray(shared_secret), AES.MODE_CBC, bytearray(iv_bytes))
        decrypted_bytes = cipher.decrypt(bytearray(encrypted_bytes))
        plaintext = decrypted_bytes.decode("utf-8")
        decoded = plaintext.rsplit("}", 1)[0] + "}"  # weird symbols at the end
        return decoded
    except Exception as ex:
        return str(ex)


# DATABASE LOGIC
def create_sql_table():
    try:
        con = sqlite3.connect(DVMConfig.USERDB)
        cur = con.cursor()
        cur.execute(""" CREATE TABLE IF NOT EXISTS users (
                                            npub text PRIMARY KEY,
                                            sats integer NOT NULL,
                                            iswhitelisted boolean,
                                            isblacklisted boolean,
                                            nip05 text,
                                            lud16 text,
                                            name text,
                                            lastactive integer
                                        ); """)
        cur.execute("SELECT name FROM sqlite_master")
        con.close()

    except Error as e:
        print(e)


def add_sql_table_column():
    try:
        con = sqlite3.connect(DVMConfig.USERDB)
        cur = con.cursor()
        cur.execute(""" ALTER TABLE users ADD COLUMN lastactive 'integer' """)
        con.close()
    except Error as e:
        print(e)


def add_to_sql_table(npub, sats, iswhitelisted, isblacklisted, nip05, lud16, name, lastactive):
    try:
        con = sqlite3.connect(DVMConfig.USERDB)
        cur = con.cursor()
        data = (npub, sats, iswhitelisted, isblacklisted, nip05, lud16, name, lastactive)
        cur.execute("INSERT INTO users VALUES(?, ?, ?, ?, ?, ?, ?, ?)", data)
        con.commit()
        con.close()
    except Error as e:
        print(e)


def update_sql_table(npub, sats, iswhitelisted, isblacklisted, nip05, lud16, name, lastactive):
    try:
        con = sqlite3.connect(DVMConfig.USERDB)
        cur = con.cursor()
        data = (sats, iswhitelisted, isblacklisted, nip05, lud16, name, lastactive, npub)

        cur.execute(""" UPDATE users
                  SET sats = ? ,
                      iswhitelisted = ? ,
                      isblacklisted = ? ,
                      nip05 = ? ,
                      lud16 = ? ,
                      name = ? ,
                      lastactive = ?
                  WHERE npub = ?""", data)
        con.commit()
        con.close()
    except Error as e:
        print(e)


def get_from_sql_table(npub):
    try:
        con = sqlite3.connect(DVMConfig.USERDB)
        cur = con.cursor()
        cur.execute("SELECT * FROM users WHERE npub=?", (npub,))
        row = cur.fetchone()
        con.close()
        return row

    except Error as e:
        print(e)


def delete_from_sql_table(npub):
    try:
        con = sqlite3.connect(DVMConfig.USERDB)
        cur = con.cursor()
        cur.execute("DELETE FROM users WHERE npub=?", (npub,))
        con.commit()
        con.close()
    except Error as e:
        print(e)


def clear_db():
    try:
        con = sqlite3.connect(DVMConfig.USERDB)
        cur = con.cursor()
        cur.execute("SELECT * FROM users WHERE npub IS NULL OR npub = '' ")
        rows = cur.fetchall()
        for row in rows:
            print(row)
            delete_from_sql_table(row[0])
        con.close()
        return rows
    except Error as e:
        print(e)


def list_db():
    try:
        con = sqlite3.connect(DVMConfig.USERDB)
        cur = con.cursor()
        cur.execute("SELECT * FROM users ORDER BY sats DESC")
        rows = cur.fetchall()
        for row in rows:
            print(row)
        con.close()
    except Error as e:
        print(e)


def update_user_balance(sender, sats):
    user = get_from_sql_table(sender)
    if user is None:
        add_to_sql_table(sender, (sats + DVMConfig.NEW_USER_BALANCE), False, False,
                         None, None, None, Timestamp.now().as_secs())
        print("NEW USER: " + sender + " Zap amount: " + str(sats) + " Sats.")
    else:
        user = get_from_sql_table(sender)
        print(sats)
        update_sql_table(sender, (user[1] + sats), user[2], user[3], user[4], user[5], user[6],
                         Timestamp.now().as_secs())
        print("UPDATE USER BALANCE: " + user[6] + " Zap amount: " + str(sats) + " Sats.")


def get_or_add_user(sender):
    user = get_from_sql_table(sender)
    if user is None:
        add_to_sql_table(sender, DVMConfig.NEW_USER_BALANCE, False, False, None,
                         None, None, Timestamp.now().as_secs())
        user = get_from_sql_table(sender)
    return user


# ADMINISTRARIVE DB MANAGEMENT
def admin_make_database_updates():
    # This is called on start of Server, Admin function to manually whitelist/blacklist/add balance/delete users
    # List all entries, why not.
    listdatabase = False
    deleteuser = False
    whitelistuser = False
    unwhitelistuser = False
    blacklistuser = False
    addbalance = False

    if listdatabase:
        list_db()

    #publickey = PublicKey.from_bech32("npub19jkj3lf4gh53qnp70uupvv3k3pyl39fzu52ygkhhszd5083yd36qpyu0dy").to_hex()
    # use this if you have the npub
    publickey = "99bb5591c9116600f845107d31f9b59e2f7c7e09a1ff802e84f1d43da557ca64"
    # publickey = "c63c5b4e21b9b1ec6b73ad0449a6a8589f6bd8542cabd9e5de6ae474b28fe806"

    if whitelistuser:
        user = get_from_sql_table(publickey)
        update_sql_table(user[0], user[1], True, False, user[4], user[5], user[6], user[7])

    if unwhitelistuser:
        user = get_from_sql_table(publickey)
        update_sql_table(user[0], user[1], False, False, user[4], user[5], user[6], user[7])

    if blacklistuser:
        user = get_from_sql_table(publickey)
        update_sql_table(user[0], user[1], False, True, user[4], user[5], user[6], user[7])

    if addbalance:
        additional_balance = 250
        user = get_from_sql_table(publickey)
        update_sql_table(user[0], user[1] + additional_balance, user[2], user[3], user[4], user[5], user[6], user[7])

    if deleteuser:
        delete_from_sql_table(publickey)

def nip89_announce_tasks():

    k65002_tag = Tag.parse(["k", 65002])
    k65003_tag = Tag.parse(["k", 65003])
    k65004_tag = Tag.parse(["k", 65004])
    k65005_tag = Tag.parse(["k", 65005])
    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    event = EventBuilder(31990, "", [k65002_tag, k65003_tag ,k65004_tag, k65005_tag]).to_event(keys)
    send_event(event)




if __name__ == '__main__':
    os.environ["NOVA_DATA_DIR"] = "W:\\nova\\data"
    os.environ["NOVA_NOSTR_KEY"] = "privkey"
    os.environ["NOVA_HOST"] = "127.0.0.1"
    os.environ["NOVA_PORT"] = "27017"
    nostr_server()
