import json

import os
import re
import urllib
from dataclasses import dataclass
from datetime import timedelta
from sqlite3 import Error
from urllib.parse import urlparse
from bech32 import bech32_decode, convertbits

import requests
import emoji
import ffmpegio

from Crypto.Cipher import AES

from decord import AudioReader, cpu
from nostr_sdk import PublicKey, Keys, Client, Tag, Event, EventBuilder, Filter, HandleNotification, Timestamp, \
    nip04_decrypt, EventId, Metadata, nostr_sdk
import time

from nova_server.route.predict_static import LLAMA2
from nova_server.utils.db_utils import db_entry_exists, add_new_session_to_db
from nova_server.utils.mediasource_utils import download_podcast, downloadYouTube
from configparser import ConfigParser
import sqlite3


# TODO
# check expiry of tasks/available output format/model/ (task is checked already). if not available ignore the job,
# send reaction on error (send sats back ideally, find out with lib, same for under payment),
# send reaction processing-scheduled when task is waiting for previous task to finish, max limit to wait?
# store whitelist (and maybe a blacklist) in a config/db
# clear list of  tasks (job_list) to watch after some time (timeout if invoice not paid),
# consider max-sat amount at all,
# consider reactions from customers (Kind 65000 event)
# add more output formats (webvtt, srt)
# purge database and files from time to time?
# Show preview of longer transcriptions, then ask for zap


class DVMConfig:
    # SUPPORTED_TASKS = ["inactive-following"]
    SUPPORTED_TASKS = ["speech-to-text", "summarization", "translation", "text-to-image", "image-to-image",
                       "image-upscale", "chat", "image-to-text"]
    LNBITS_INVOICE_KEY = 'bfdfb5ecfc0743daa08749ce58abea74'
    LNBITS_INVOICE_URL = 'https://ln.novaannotation.com/createLightningInvoice'
    USERDB = "W:\\nova\\tools\\AnnoDBbackup\\nostrzaps.db"
    RELAY_LIST = ["wss://relay.damus.io", "wss://blastr.f7z.xyz", "wss://nostr-pub.wellorder.net", "wss://nos.lol",
                  "wss://nostr.wine", "wss://relay.nostr.com.au", "wss://relay.snort.social"]
    RELAY_TIMEOUT = 1
    AUTOPROCESS_MIN_AMOUNT: int = 1000000000000  # auto start processing if min Sat amount is given
    AUTOPROCESS_MAX_AMOUNT: int = 0  # if this is 0 and min is very big, autoprocess will not trigger
    SHOWRESULTBEFOREPAYMENT: bool = True  # if this is true show results even when not paid right after autoprocess
    NEW_USER_BALANCE: int = 250  # Free credits for new users
    COSTPERUNIT_TRANSLATION: int = 20  # Still need to multiply this by duration
    COSTPERUNIT_SPEECHTOTEXT: int = 100  # Still need to multiply this by duration
    COSTPERUNIT_IMAGEGENERATION: int = 50  # Generate / Transform one image
    COSTPERUNIT_IMAGETRANSFORMING: int = 30  # Generate / Transform one image
    COSTPERUNIT_IMAGEUPSCALING: int = 25  # This takes quite long..
    COSTPERUNIT_INACTIVE_FOLLOWING: int = 250  # This takes quite long..
    COSTPERUNIT_OCR: int = 20
    REQUIRES_NIP05: bool = False
    PASSIVE_MODE: bool = False  # Run this if this instance should only do tasks set in SUPPORTED_TASKS, no chatting,
    # zap handling etc.


@dataclass
class JobToWatch:
    id: str
    timestamp: str
    is_paid: bool
    amount: int
    status: str
    result: str
    is_processed: bool


job_list = []


# init_logger(LogLevel.DEBUG)

def nostr_server():
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
            print(f"[Nostr] Received new event from {relay_url}: {event.as_json()}")
            if 65002 <= event.kind() <= 66000:
                user = get_or_add_user(event.pubkey().to_hex())
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
                else:
                    print("[Nostr] Got new Task but can't process it, skipping..")
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
                                print(f"Name: {name}")
                                print(f"NIP05: {nip05}")
                                print(f"LUD16: {lud16}")

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
                        print(f"Received new msg: {dec_text}")
                        task = str(dec_text).split(' ')[0].removeprefix('-')
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
                            print("Request from " + name + " (" + nip05 + ") Task: " + task)
                            tags.append(Tag.parse(["p", event.pubkey().to_hex()]))
                            evt = EventBuilder(4, "", tags).to_event(keys)
                            print(evt.as_json())
                            do_work(evt, is_from_bot=True)

                        else:
                            time.sleep(3.0)
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(),
                                "Balance required, please zap this note with at least " + str(required_amount)
                                + " Sats to start directly, or zap me that amount elsewhere, then try again.",
                                event.id()).to_event(keys)
                            job_list.append(
                                JobToWatch(id=evt.id().to_hex(), timestamp=str(event.created_at().as_secs()),
                                           amount=required_amount, is_paid=False, status="payment-required", result="",
                                           is_processed=False))
                            send_event(evt, client)

                    elif not DVMConfig.PASSIVE_MODE:
                        print("Request from " + name + " (" + nip05 + ") Message: " + dec_text)
                        if str(dec_text).startswith("-balance"):
                            get_or_add_user(sender)
                            balance = user[1]
                            time.sleep(3.0)
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(),
                            "Your current balance is " + str(balance) + " Sats. Zap me to add to your balance. "
                            "I support both public and private Zaps, as well as Zapplepay.", None).to_event(keys)
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
                        else:
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(), "I dont understand"
                                  " the command. Please type -help to see what I can do for you.",
                                   event.id()).to_event(keys)
                            send_event(evt, client)

                except Exception as e:
                    print(f"Error during content decryption: {e}")
            elif event.kind() == 9734:
                print(event.as_json())
            elif event.kind() == 9735 and not DVMConfig.PASSIVE_MODE:
                print(event.as_json())
                print("Zap received")
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
                    if zapped_event is not None:
                        if zapped_event.kind() == 65000:  # if a reaction by us got zapped
                            amount = 0
                            job_event = None
                            for tag in zapped_event.tags():
                                if tag.as_vec()[0] == 'amount':
                                    amount = int(float(tag.as_vec()[1]) / 1000)
                                elif tag.as_vec()[0] == 'e':
                                    job_event = get_event_by_id(tag.as_vec()[1])
                                    print("[Nostr] Original Job Request event found...")

                            if job_event is not None:
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
                        elif zapped_event.kind() == 4:
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
                        elif not anon:
                            update_user_balance(sender, invoice_amount)
                            # a regular note
                    elif not anon:
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
                        except Exception:
                            try:
                                t = time.strptime(tag.as_vec()[2], "%M:%S")
                                seconds = t.tm_min * 60 + t.tm_sec
                                request_form["startTime"] = str(seconds)
                            except Exception:
                                request_form["startTime"] = tag.as_vec()[2]
                        try:
                            t = time.strptime(tag.as_vec()[3], "%H:%M:%S")
                            seconds = t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec
                            request_form["endTime"] = str(seconds)
                        except Exception:
                            try:
                                t = time.strptime(tag.as_vec()[3], "%M:%S")
                                seconds = t.tm_min * 60 + t.tm_sec
                                request_form["endTime"] = str(seconds)
                            except Exception:
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
            # outsource this to its own script, ideally. This is not using the database for now, but probably should.
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
                    if tag.as_vec()[1] == "prompt":  # check for paramtype
                        prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "negative_prompt":  # check for paramtype
                        negative_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "strength":  # check for paramtype
                        strength = float(tag.as_vec()[2])
                    elif tag.as_vec()[1] == "guidance_scale":  # check for paramtype
                        guidance_scale = float(tag.as_vec()[2])
                    elif tag.as_vec()[1] == "model":  # check for paramtype
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
                    if tag.as_vec()[1] == "prompt":  # check for paramtype
                        extra_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "negative_prompt":  # check for paramtype
                        negative_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "size":  # check for paramtype
                        width = tag.as_vec()[2]
                        height = tag.as_vec()[3]
                    elif tag.as_vec()[1] == "ratio":  # check for paramtype
                        ratio_width = (tag.as_vec()[2])
                        ratio_height = (tag.as_vec()[3])
                    elif tag.as_vec()[1] == "upscale":  # check for paramtype
                        upscale = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "model":  # check for paramtype
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
                    if tag.as_vec()[1] == "upscale":  # check for paramtype
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

            # add length variableF
        elif task == "summarization":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = 'summarization'
            text = ""
            user = event.pubkey().to_hex()
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    input_type = tag.as_vec()[2]
                    if input_type == "url":
                        text = tag.as_vec()[1].replace(";", "")
                    elif input_type == "event":
                        evt = get_event_by_id(tag.as_vec()[1])
                        text = evt.content().replace(";", "")
                    elif input_type == "job":
                        job_id_filter = Filter().kind(65001).event(EventId.from_hex(tag.as_vec()[1])).limit(1)
                        events = client.get_events_of([job_id_filter], timedelta(seconds=DVMConfig.RELAY_TIMEOUT))
                        if len(events) > 0:
                            evt = events[0]
                            text = evt.content().replace(";", "")
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

            # is youtube link?
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
                if organize_input_data(job_event, request_form) is not None:
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
def check_event_status(content, originaleventstr: str, use_bot=False):
    originalevent = Event.from_json(originaleventstr)
    for x in job_list:
        if x.id == originalevent.id().to_hex():
            is_paid = x.is_paid
            amount = x.amount
            x.result = content
            x.is_processed = True
            if DVMConfig.SHOWRESULTBEFOREPAYMENT and not is_paid:
                send_nostr_reply_event(content, originaleventstr)
                send_job_status_reaction(originalevent, "success", amount)  # or payment-required, or both?
            elif not DVMConfig.SHOWRESULTBEFOREPAYMENT and not is_paid:
                send_job_status_reaction(originalevent, "success", amount)  # or payment-required, or both?

            if DVMConfig.SHOWRESULTBEFOREPAYMENT and is_paid:
                job_list.remove(x)
            elif not DVMConfig.SHOWRESULTBEFOREPAYMENT and is_paid:
                job_list.remove(x)
                send_nostr_reply_event(content, originaleventstr)
            print(str(job_list))
            break

    else:
        resultcontent = post_process_result(content, originalevent)
        print(str(job_list))
        if use_bot:
            keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
            receiver_key = PublicKey()
            for tag in originalevent.tags():
                if tag.as_vec()[0] == "p":
                    receiver_key = PublicKey.from_hex(tag.as_vec()[1])
            event = EventBuilder.new_encrypted_direct_msg(keys, receiver_key, resultcontent, None).to_event(keys)
            send_event(event)

        else:
            send_nostr_reply_event(resultcontent, originaleventstr)
            send_job_status_reaction(originalevent, "success")


# NIP90 REPLIES
def respond_to_error(content, originaleventstr, is_from_bot=False):
    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    originalevent = Event.from_json(originaleventstr)
    sender = ""
    task = ""
    if not is_from_bot:
        send_job_status_reaction(originalevent, "error", content=content)
        # TODO Send Zap back
    else:
        for tag in originalevent.tags():
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
    altdesc = "This is a reaction to a NIP90 DVM AI task."
    task = get_task(original_event)
    if status == "processing":
        reaction = emoji.emojize(":thumbs_up:")
        altdesc = "NIP90 DVM AI task " + task + " started processing."
    elif status == "success":
        reaction = emoji.emojize(":call_me_hand:")
        altdesc = "NIP90 DVM AI task " + task + " finished successfully."
    elif status == "error":
        if content is None:
            reaction = emoji.emojize(":thumbs_down:")
        else:
            reaction = emoji.emojize(":thumbs_down:") + content
        altdesc = "NIP90 DVM AI task " + task + (" had an error. So sorry. In the future zaps will be sent back"
                                                 " but I can't do that just yet.")
    elif status == "payment-required":
        reaction = emoji.emojize(":orange_heart:")
        altdesc = "NIP90 DVM AI task " + task + " requires payment of min " + str(amount) + " Sats."
        if task == "speech-to-text":
            altdesc = altdesc + (" Providing results with WhisperX large-v2. "
                                 "Accepted input formats: wav,mp3,mp4,ogg,avi,mov,youtube,overcast. "
                                 "Possible outputs: text/plain, timestamped labels depending on "
                                 "alignment parameter (word,segment,raw)")
        elif task == "image-to-text":
            altdesc = altdesc + (" Accepted input formats: jpg. Possible outputs: text/plain. "
                                 "This is very experimental, make sure your text is well readable.")

    elif status == "payment-rejected":
        reaction = emoji.emojize(":see_no_evil_monkey:")
        altdesc = "NIP90 DVM AI task " + task + " payment is below required amount of " + str(amount) + " Sats."
    elif status == "user-blocked-from-service":
        reaction = emoji.emojize(":see_no_evil_monkey:")
        altdesc = "NIP90 DVM AI task " + task + " can't be performed. User has been blocked from Service"
    else:
        reaction = emoji.emojize(":see_no_evil_monkey:")

    etag = Tag.parse(["e", original_event.id().to_hex()])
    ptag = Tag.parse(["p", original_event.pubkey().to_hex()])
    alttag = Tag.parse(["alt", altdesc])
    statustag = Tag.parse(["status", status])
    tags = [etag, ptag, alttag, statustag]

    if status == "success" or status == "error":  #
        for x in job_list:
            if x.id == original_event.id():
                is_paid = x.is_paid
                amount = x.amount
                break
    if status == "payment-required" or (status == "processing" and not is_paid):
        job_list.append(
            JobToWatch(id=original_event.id().to_hex(), timestamp=original_event.created_at().as_secs(), amount=amount,
                       is_paid=is_paid,
                       status=status, result="", is_processed=False))
        print(str(job_list))
    if status == "payment-required" or status == "payment-rejected" or (status == "processing" and not is_paid) or (
            status == "success" and not is_paid):
        # try:
        #    if DVMConfig.LNBITS_INVOICE_KEY != "":
        #        bolt11 = createBolt11LnBits(amount)
        #        amount_tag = Tag.parse(["amount", str(amount), bolt11])
        # except Exception as e:
        #   print(e)

        amount_tag = Tag.parse(["amount", str(amount * 1000)])  # to millisats
        tags.append(amount_tag)

    keys = Keys.from_sk_str(os.environ["NOVA_NOSTR_KEY"])
    event = EventBuilder(65000, reaction, tags).to_event(keys)

    send_event(event, client)
    print("[Nostr] Sent Kind 65000 Reaction: " + status + " " + event.as_json())
    return event.as_json()


# POSTPROCESSING
def post_process_result(content, original_event):
    for tag in original_event.tags():
        if tag.as_vec()[0] == "output":
            if tag.as_vec()[1] == "text/plain":
                result = ""
                try:
                    for name in content['name']:
                        cleared_name = str(name).lstrip("\'").rstrip("\'")
                        result = result + cleared_name + "\n"
                    content = str(result).replace("\"", "").replace('[', "").replace(']', "").lstrip(None)
                except Exception as e:
                    print("Can't transform text, or text already in text/plain format. " + str(e))
            # TODO add more

    return content


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
        prompttemp = dec_text.replace("-text-to-image ", "")
        split = prompttemp.split(" -")
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
        prompttemp = dec_text.replace("-image-to-image ", "")
        split = prompttemp.split(" -")
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
        prompttemp = dec_text.replace("-image-upscale ", "")
        split = prompttemp.split(" -")
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
        prompttemp = dec_text.replace("-image-to-text ", "")
        split = prompttemp.split(" -")
        url = str(split[0]).replace(' ', '')
        j_tag = Tag.parse(["j", "image-to-text"])
        i_tag = Tag.parse(["i", url, "url"])
        tags = [j_tag, i_tag]
        return tags

    elif str(dec_text).startswith("-speech-to-text"):
        prompttemp = dec_text.replace("-speech-to-text ", "")
        split = prompttemp.split(" -")
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
        sincedays = "30"
        numberusers = "25"
        prompttemp = dec_text.replace("-inactive-following ", "")
        print(prompttemp)
        split = prompttemp.split(" -")
        for i in split:
            if i.startswith("sincedays "):
                sincedays = i.replace("sincedays ", "")
            elif i.startswith("num"):
                numberusers = i.replace("num ", "")
        param_tag_since = Tag.parse(["param", "since", sincedays])
        param_tag_num_users = Tag.parse(["param", "numusers", numberusers])
        j_tag = Tag.parse(["j", "inactive-following"])
        return [j_tag, param_tag_since, param_tag_num_users]
    else:
        text = dec_text
        j_tag = Tag.parse(["j", "chat"])
        i_tag = Tag.parse(["i", text, "text"])
        return [j_tag, i_tag]


# CHECK INPUTS/TASK AVAILABLE
def check_task_is_supported(event):
    task = get_task(event)
    input_value = ""
    input_type = ""
    print("Received new Task: " + task)
    has_i_tag = False
    for tag in event.tags():
        if tag.as_vec()[0] == 'i':
            if len(tag.as_vec()) < 2:
                has_i_tag = False
            else:
                input_value = tag.as_vec()[1]
                input_type = tag.as_vec()[2]
                has_i_tag = True
        elif tag.as_vec()[0] == 'output':
            output = tag.as_vec()[1]
            if output != "text/plain":
                return False

    if not has_i_tag:
        return False

    if task not in DVMConfig.SUPPORTED_TASKS:  # The Tasks this DVM supports (can be extended)
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
    remaininginvoice = invoice[4:]
    index = get_index_of_first_letter(remaininginvoice)
    identifier = remaininginvoice[index]
    numberstring = remaininginvoice[:index]
    number = float(numberstring)
    if identifier == 'm':
        number = number * 100000000 * 0.001
    elif identifier == 'u':
        number = number * 100000000 * 0.000001
    elif identifier == 'n':
        number = number * 100000000 * 0.000000001
    elif identifier == 'p':
        number = number * 100000000 * 0.000000000001

    return int(number)


def create_bolt11_ln_bits(millisats):
    sats = int(millisats / 1000)
    url = DVMConfig.LNBITS_INVOICE_URL
    data = {'invoice_key': DVMConfig.LNBITS_INVOICE_KEY, 'sats': str(sats), 'memo': "Nostr-DVM"}
    res = requests.post(url, data=data)
    obj = json.loads(res.text)
    return obj["payment_request"]


def get_index_of_first_letter(ip):
    index = 0
    for c in ip:
        if c.isalpha():
            return index
        else:
            index = index + 1
    return len(ip)


# DECRYPTZAPS
def check_for_zapplepay(sender, content):
    try:
        # Special case Zapplepay
        if sender == PublicKey.from_bech32("npub1wxl6njlcgygduct7jkgzrvyvd9fylj4pqvll6p32h59wyetm5fxqjchcan").to_hex():
            # '71bfa9cbf84110de617e959021b08c69524fcaa1033ffd062abd0ae2657ba24c' # Just for sanity, Zapplepay hexkey
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
        print("NEW USER")
    else:
        user = get_from_sql_table(sender)
        print(sats)
        update_sql_table(sender, (user[1] + sats), user[2], user[3], user[4], user[5], user[6],
                         Timestamp.now().as_secs())
        print("UPDATE USER BALANCE")


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

    publickey = PublicKey.from_bech32(
        "npub1cc79kn3phxc7c6mn45zynf4gtz0khkz59j4anew7dtj8fv50aqrqlth2hf").to_hex()  # use this if you have the npub
    # publickey = "99bb5591c9116600f845107d31f9b59e2f7c7e09a1ff802e84f1d43da557ca64"
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


if __name__ == '__main__':
    os.environ["NOVA_DATA_DIR"] = "W:\\nova\\data"
    os.environ["NOVA_NOSTR_KEY"] = "privkey"
    os.environ["NOVA_HOST"] = "127.0.0.1"
    os.environ["NOVA_PORT"] = "27017"
    nostr_server()
