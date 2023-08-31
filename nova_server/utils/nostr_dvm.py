import json

import os
import re
import urllib
from dataclasses import dataclass
from datetime import timedelta
from sqlite3 import Error
from types import NoneType
from urllib.parse import urlparse

from bech32 import bech32_decode, convertbits

import requests
import emoji
import ffmpegio

from Crypto.Cipher import AES

from decord import AudioReader, cpu
from nostr_sdk import PublicKey, Keys, Client, Tag, Event, EventBuilder, Filter, HandleNotification, Timestamp, \
    nip04_decrypt, EventId, Metadata, nostr_sdk, Alphabet, ClientMessage, Options

import time

from nova_utils.ssi_utils.ssi_anno_utils import Anno

from nova_server.route.predict_static import LLAMA2
from nova_server.utils.db_utils import db_entry_exists, add_new_session_to_db
from nova_server.utils.mediasource_utils import download_podcast, downloadYouTube
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

class DVMConfig:
    SUPPORTED_TASKS = [] # ["inactive-following", "note-recommendation", "speech-to-text", "summarization", "translation", "text-to-image", "image-to-image", "image-upscale","image-to-text", "image-reimagine"]
    PRIVATE_KEY: str


    IS_HYBRID: bool = False  # Once instance serves as both bot, and dvm
    IS_BOT: bool = False  # This should act as Bot, not NIP90 DVM
    PASSIVE_MODE: bool = True  # Do not listen to zaps (So they are not registered twice, when in bot mode


    USERDB = "W:\\nova\\tools\\AnnoDBbackup\\nostrzaps.db"
    RELAY_LIST = ["wss://relay.damus.io", "wss://blastr.f7z.xyz", "wss://nostr-pub.wellorder.net", "wss://nos.lol",
                  "wss://nostr.wine", "wss://relay.nostr.com.au", "wss://relay.snort.social"]
    RELAY_TIMEOUT = 1
    LNBITS_INVOICE_KEY = "" # 'bfdfb5ecfc0743daa08749ce58abea74'
    LNBITS_URL = 'https://lnbits.novaannotation.com'
    REQUIRES_NIP05: bool = False

    AUTOPROCESS_MIN_AMOUNT: int = 1000000000000  # auto start processing if min Sat amount is given
    AUTOPROCESS_MAX_AMOUNT: int = 0  # if this is 0 and min is very big, autoprocess will not trigger
    SHOWRESULTBEFOREPAYMENT: bool = True  # if this is true show results even when not paid right after autoprocess
    NEW_USER_BALANCE: int = 250  # Free credits for new users

    COSTPERUNIT_TRANSLATION: int = 20  # Still need to multiply this by duration
    COSTPERUNIT_SPEECHTOTEXT: float = 0.1  # Still need to multiply this by duration
    COSTPERUNIT_IMAGEGENERATION: int = 50  # Generate / Transform one image
    COSTPERUNIT_IMAGETRANSFORMING: int = 50  # Generate / Transform one image
    COSTPERUNIT_IMAGEUPSCALING: int = 25  # This takes quite long..
    COSTPERUNIT_INACTIVE_FOLLOWING: int = 500  # This takes quite long..
    COSTPERUNIT_NOTE_RECOMMENDATION: int = 100  # testing
    COSTPERUNIT_OCR: int = 20
    NIP89s: list = []

class NIP89Announcement:
    kind: int
    dtag: str
    pk: str
    content: str


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

@dataclass
class RequiredJobToWatch:
    event: Event
    timestamp: int



job_list = []
jobs_on_hold_list = []
dvmconfig = DVMConfig()

# init_logger(LogLevel.DEBUG)
def nostr_server(config):
    dvmconfig = config

    keys = Keys.from_sk_str(dvmconfig.PRIVATE_KEY)
    sk = keys.secret_key()
    pk = keys.public_key()

    if dvmconfig.IS_HYBRID:
        print(f"Nostr DVM/Bot hybrid public key: {pk.to_bech32()}, Hex: {pk.to_hex()} ")
        print(f"Supported tasks: {dvmconfig.SUPPORTED_TASKS}")
    elif dvmconfig.IS_BOT:
        print(f"Nostr Bot public key: {pk.to_bech32()}, Hex: {pk.to_hex()} ")
        print(f"Supported Bot tasks: {dvmconfig.SUPPORTED_TASKS}")
    else:
        print(f"Nostr DVM public key: {pk.to_bech32()}, Hex: {pk.to_hex()} ")
        print(f"Supported DVM tasks: {dvmconfig.SUPPORTED_TASKS}")


    client = Client(keys)
    for relay in dvmconfig.RELAY_LIST:
        client.add_relay(relay)
    client.connect()

    if dvmconfig.IS_HYBRID:
        kinds = [4]
        if not dvmconfig.PASSIVE_MODE:
            kinds = [4, 9735]

        dm_zap_filter = Filter().pubkey(pk).kinds(kinds).since(Timestamp.now())
        dvm_filter = (Filter().kinds([66000, 65002, 65003, 65004, 65005, 65007]).since(Timestamp.now()))
        client.subscribe([dm_zap_filter, dvm_filter])

    elif dvmconfig.IS_BOT:
        kinds = [4]
        if not dvmconfig.PASSIVE_MODE:
            kinds = [4, 9735]

        dm_zap_filter = Filter().pubkey(pk).kinds(kinds).since(Timestamp.now())
        client.subscribe([dm_zap_filter])

    else:
        dm_zap_filter = Filter().pubkey(pk).kinds([9735]).since(Timestamp.now())
        dvm_filter = (Filter().kinds([66000, 65002, 65003, 65004, 65005, 65007]).since(Timestamp.now()))
        client.subscribe([dm_zap_filter, dvm_filter])

    create_sql_table()
    admin_make_database_updates(config=dvmconfig)

    class NotificationHandler(HandleNotification):
        def handle(self, relay_url, event):
            if 65002 <= event.kind() <= 66000:
                print(f"[Nostr] Received new NIP90 Job Request from {relay_url}: {event.as_json()}")
                if check_event_has_not_unifinished_job_input(event,True, client, dvmconfig):
                    handle_nip90_job_event(event)

            elif event.kind() == 4:
                sender = event.pubkey().to_hex()
                try:
                    dec_text = nip04_decrypt(sk, event.pubkey(), event.content())
                    user = get_or_add_user(sender)
                    nip05 = user[4]
                    name = user[6]
                    if nip05 == None:
                        nip05 = ""
                    if name == None:
                        name = ""
                    # Get nip05,lud16 and name from profile and store them in db.
                    if str(nip05) == "" or str(name) == "":
                        try:
                            profile_filter = Filter().kind(0).author(event.pubkey().to_hex()).limit(1)
                            events = client.get_events_of([profile_filter], timedelta(seconds=3))
                            if len(events) > 0:
                                ev = events[0]
                                metadata = Metadata.from_json(ev.content())
                                name = metadata.get_display_name()
                                if str(name) == "" or name is None:
                                    name = metadata.get_name()
                                nip05 = metadata.get_nip05()
                                lud16 = metadata.get_lud16()
                                update_sql_table(user[0], user[1], user[2], user[3], nip05, lud16, name,
                                                 Timestamp.now().as_secs())
                                user = get_from_sql_table(user[0])
                                if str(nip05) == "" or nip05 is None:

                                    if dvmconfig.REQUIRES_NIP05 and int(user[1]) <= dvmconfig.NEW_USER_BALANCE:
                                        time.sleep(1.0)
                                        message = (("In order to reduce misuse by bots, a NIP05 address or a balance "
                                                    "higher than the free credits (") + str(dvmconfig.NEW_USER_BALANCE)
                                                   + " Sats) is required to use this service. You can zap any of my "
                                                     "notes or my profile using public or private zaps. "
                                                     "Zapplepay is also supported")

                                        evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(), message,
                                                                                    event.id()).to_event(keys)
                                        send_event(evt, key=keys)
                                        return
                        except Exception as e:
                            if not user[2]: #whitelisted
                                amount = int(user[1]) + get_amount_per_task(get_task(event, client), config=dvmconfig)
                                update_sql_table(sender, amount, user[2], user[3], user[4],
                                                 user[5], user[6],
                                                 Timestamp.now().as_secs())
                                message = "There was the following error : " + str(e) + ". Credits have been reimbursed"
                            else:
                                # User didn't pay, so no reimbursement
                                message = "There was the following error : " + str(e)

                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(), message, event.id()).to_event(keys)
                            send_event(evt, key=keys)
                            print(e)

                    # update last active status
                    update_sql_table(user[0], user[1], user[2], user[3], user[4], user[5], user[6],
                                     Timestamp.now().as_secs())
                    if any(dec_text.startswith("-" + s) for s in dvmconfig.SUPPORTED_TASKS):
                        task = str(dec_text).split(' ')[0].removeprefix('-')
                        print("Request from " + str(name) + " (" + str(nip05) + ") Task: " + str(task))
                        required_amount = get_amount_per_task(task, config=dvmconfig)
                        balance = int(user[1])
                        is_whitelisted = user[2]
                        is_blacklisted = user[3]
                        if is_blacklisted:
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(),
                                                                "Your are currently blocked from all services.",
                                                                None).to_event(keys)
                            send_event(evt, key=keys)
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

                            print("Replying with scheduled confirmation")
                            send_event(evt, key=keys)

                            #build temp event to work with
                            tags = parse_bot_command_to_event(dec_text, event.pubkey().to_hex())
                            tags.append(Tag.parse(["p", event.pubkey().to_hex()]))
                            jobevt = EventBuilder(4, "", tags).to_event(keys)


                            #expires = event.created_at().as_secs() + (60 * 60)
                            #job_list.append(
                            #    JobToWatch(event_id=evt.id().to_hex(), timestamp=event.created_at().as_secs(),
                            #               amount=required_amount, is_paid=True, status="processing", result="",
                            #               is_processed=False, bolt11="", payment_hash="", expires=expires, from_bot=True))
                            print("Do work..")
                            do_work(jobevt, is_from_bot=True)




                        else:
                            print("payment-required")
                            time.sleep(3.0)
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(),
                                "Balance required, please zap me with at least " + str(required_amount)
                                + " Sats, then try again.",
                                event.id()).to_event(keys)
                            send_event(evt, key=keys)

                    elif not dvmconfig.PASSIVE_MODE:
                        print("Request from " + str(name) + " (" +str (nip05) + ") Message: " + dec_text)
                        if str(dec_text).startswith("-balance"):
                            user = get_or_add_user(sender)
                            balance = int(user[1])
                            time.sleep(3.0)
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(),
                                "Your current balance is " + str(balance) + " Sats. Zap me to add to your balance. "
                                "I support both public and private Zaps, as well as Zapplepay.",
                                 None).to_event(keys)
                            send_event(evt, key=keys)
                        elif str(dec_text).startswith("-help") or str(dec_text).startswith("- help") or str(
                                dec_text).startswith("help"):
                            time.sleep(3.0)
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(), get_bot_help_text(),
                                                                        event.id()).to_event(keys)
                            send_event(evt, key=keys)
                        elif str(dec_text).lower().__contains__("bitcoin"):
                            time.sleep(3.0)
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(),
                                 "#Bitcoin? There is no second best.\n\nhttps://cdn.nostr.build/p/mYLv.mp4",
                                  event.id()).to_event(keys)
                            send_event(evt, key=keys)
                        elif not str(dec_text).startswith("-"):
                            # Contect LLAMA Server in parallel to cue.
                            answer = LLAMA2(dec_text, event.pubkey().to_hex())
                            evt = EventBuilder.new_encrypted_direct_msg(keys, event.pubkey(), answer,
                                                                        event.id()).to_event(keys)
                            send_event(evt, key=keys)

                except Exception as e:
                    print(f"Error during content decryption: {e}")
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
                            zapped_event = get_event_by_id(tag.as_vec()[1], config=dvmconfig)
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
                    print(str(user))
                    print("Zap received: " + str(invoice_amount) + " Sats from " + str(user[6]))
                    if zapped_event is not None:
                        if zapped_event.kind() == 65000:  # if a reaction by us got zapped
                            amount = 0
                            job_event = None
                            for tag in zapped_event.tags():
                                if tag.as_vec()[0] == 'amount':
                                    amount = int(float(tag.as_vec()[1]) / 1000)
                                elif tag.as_vec()[0] == 'e':
                                    job_event = get_event_by_id(tag.as_vec()[1], config=dvmconfig)

                            task_supported, task, duration = check_task_is_supported(job_event, client=client, get_duration=False, config=dvmconfig)
                            if job_event is not None and task_supported:
                                if amount <= invoice_amount:
                                    print("[Nostr] Payment-request fulfilled...")
                                    send_job_status_reaction(job_event, "processing", client=client, config=dvmconfig)
                                    indices = [i for i, x in enumerate(job_list) if x.event_id == job_event.id().to_hex()]
                                    index = -1
                                    if len(indices) > 0:
                                        index = indices[0]
                                    if index > -1:
                                        if job_list[index].is_processed:  # If payment-required appears after processing
                                            job_list[index].is_paid = True
                                            check_event_status(job_list[index].result, str(job_event.as_json()), dvm_key=dvmconfig.PRIVATE_KEY)
                                        elif not (job_list[index]).is_processed:
                                            # If payment-required appears before processing
                                            job_list.pop(index)
                                            print("Starting work...")
                                            do_work(job_event, is_from_bot=False)
                                else:
                                    send_job_status_reaction(job_event, "payment-rejected",
                                                             False, invoice_amount, client=client, config=dvmconfig)
                                    print("[Nostr] Invoice was not paid sufficiently")

                        elif zapped_event.kind() == 65001:
                            print("Someone zapped the result of an exisiting Task. Nice")
                        elif not anon and not dvmconfig.PASSIVE_MODE:
                            update_user_balance(sender, invoice_amount,config=dvmconfig)

                            # a regular note
                    elif not anon and not dvmconfig.PASSIVE_MODE:
                        update_user_balance(sender, invoice_amount,config=dvmconfig)

                except Exception as e:
                    print(f"Error during content decryption: {e}")

        def handle_msg(self, relay_url, msg):
            return



    def handle_nip90_job_event(event):
        user = get_or_add_user(event.pubkey().to_hex())
        is_whitelisted = user[2]
        is_blacklisted = user[3]
        if is_whitelisted:
            task_supported, task, duration = check_task_is_supported(event, client=client, get_duration=False, config=dvmconfig)
            print(task)
        else:
            task_supported, task, duration = check_task_is_supported(event, client=client, get_duration=True, config=dvmconfig)
            print(task)
            print(duration)
            print(task_supported)

        if is_blacklisted:
            send_job_status_reaction(event, "error", client=client, config=dvmconfig)
            print("[Nostr] Request by blacklisted user, skipped")

        elif task_supported:
            print("Received new Task: " + task)
            print(duration)
            amount = get_amount_per_task(task, duration, config=dvmconfig)
            if amount is None:
                return

            if is_whitelisted or task == "chat":
                print("[Nostr] Whitelisted for task " + task + ". Starting processing..")
                send_job_status_reaction(event, "processing", True, 0, client=client, config=dvmconfig)
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
                    if (bid_offer > dvmconfig.AUTOPROCESS_MIN_AMOUNT or
                            bid_offer < dvmconfig.AUTOPROCESS_MAX_AMOUNT):
                        print("[Nostr][Auto-processing: Payment suspended to end] Job event: " + str(
                            event.as_json()))
                        do_work(event, is_from_bot=False)
                    else:
                        if bid_offer >= amount:
                            send_job_status_reaction(event, "payment-required", False,
                                                     amount, # bid_offer
                                                     client=client, config=dvmconfig)
                        #else:
                        #    send_job_status_reaction(event, "payment-rejected", False,
                        #                             amount,
                        #                             client=client, config=dvmconfig)  # Reject and tell user minimum amount

                else:  # If there is no bid, just request server rate from user
                    print("[Nostr] Requesting payment for Event: " + event.id().to_hex())
                    send_job_status_reaction(event, "payment-required",
                                             False, amount, client=client,  config=dvmconfig)
        else:
            print("Task not supported on this DVM, skipping..")
    # PREPARE REQUEST FORM AND DATA AND SEND TO PROCESSING
    def create_requestform_from_nostr_event(event, is_bot=False):
        task = get_task(event, client=client)

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
                        "sessions": event.id().to_hex(), "isBot": str(is_bot), "dvmkey": dvmconfig.PRIVATE_KEY,
                        "startTime": "0", "endTime": "0"}

        if task == "speech-to-text":
            # Declare specific model type e.g. whisperx_large-v2
            request_form["mode"] = "PREDICT"
            alignment = "raw"
            model_option = "base" #"large-v2"

            for tag in event.tags():
                if tag.as_vec()[0] == 'param':
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
                    elif param == "lang":  # check for paramtype
                        translation_lang = str(tag.as_vec()[2]).split('-')[0]

            if input_type == "event":
                for tag in event.tags():
                    if tag.as_vec()[0] == 'i':
                        evt = get_event_by_id(tag.as_vec()[1], config=dvmconfig)
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
                        evt = get_referenced_event_by_id(tag.as_vec()[1], [65001], client, config=dvmconfig)
                        text = evt.content()
                        break

            request_form["optStr"] = 'translation_lang=' + translation_lang + ';text=' + text.replace('\U0001f919', "").replace("=", "equals").replace(";", ",")

        elif task == "image-to-text":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = task
            input_type = "url"
            url = ""
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    if input_type == "url":
                        url = tag.as_vec()[1]
                    elif input_type == "event":
                        evt = get_event_by_id(tag.as_vec()[1], config=dvmconfig)
                        url = re.search("(?P<url>https?://[^\s]+)", evt.content()).group("url")
                    elif input_type == "job":
                        evt = get_referenced_event_by_id(tag.as_vec()[1], [65001], client, config=dvmconfig)
                        url = re.search("(?P<url>https?://[^\s]+)", evt.content()).group("url")
            request_form["optStr"] = 'url=' + url

        elif task == "image-to-image":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = task
            prompt = "surprise me"
            url = " "
            negative_prompt = " "
            strength = 0.5
            guidance_scale = 7.5
            model = "sdxl"
            lora = ""

            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    input_type = tag.as_vec()[2]
                    if input_type == "url":
                        url = tag.as_vec()[1]
                    elif input_type == "text":
                        prompt = tag.as_vec()[1]
                    elif input_type == "event":
                        evt = get_event_by_id(tag.as_vec()[1], config=dvmconfig)
                        url = re.search("(?P<url>https?://[^\s]+)", evt.content()).group("url")
                    elif input_type == "job":
                         evt = get_referenced_event_by_id(tag.as_vec()[1], [65001], client, config=dvmconfig)
                         if evt is not None:
                             url = evt.content()
                elif tag.as_vec()[0] == 'param':
                    if tag.as_vec()[1] == "negative_prompt":
                        negative_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "extra_prompt":
                        prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "strength":
                        strength = float(tag.as_vec()[2])
                    elif tag.as_vec()[1] == "guidance_scale":
                        guidance_scale = float(tag.as_vec()[2])
                    elif tag.as_vec()[1] == "model":
                        model = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "lora":
                        lora = tag.as_vec()[2]

            request_form["optStr"] = ('url=' + url + ';prompt=' + prompt + ';negative_prompt=' + negative_prompt
                                     + ';strength=' + str(strength) + ';guidance_scale=' + str(guidance_scale)
                                     + ';model=' + model
                                     + ';lora=' + lora)

        elif task == "text-to-image":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = task
            prompt = ""
            extra_prompt = ""
            negative_prompt = ""
            upscale = "4"
            model = "stabilityai/stable-diffusion-xl-base-1.0"

            ratio_width = "1"
            ratio_height = "1"
            lora = ""

            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    input_type = tag.as_vec()[2]
                    if input_type == "text":
                        prompt = tag.as_vec()[1]
                    elif input_type == "event":
                        evt = get_event_by_id(tag.as_vec()[1], config=dvmconfig)
                        llamalist = LLAMA2("Give me the keywords of the following input: "  + evt.content(), "",
                                           "Reply only with comma-seperated lists, no smalltalk")
                        prompt = llamalist.replace("\n", ",")
                    elif input_type == "job":
                        evt = get_referenced_event_by_id(tag.as_vec()[1], [65001], client, config=dvmconfig)
                        if evt is not None:
                            try:
                                llamalist = LLAMA2(evt.content(), ""
                                                   ,"Give me maxium 25 keywords for the given text. Reply only with comma-seperated lists, no smalltak")
                                promptarr = llamalist.split(":")
                                if len(promptarr) > 1:
                                    prompt = promptarr[1].lstrip("\n").replace("\n", ",").replace("*", ",")
                                else:
                                    prompt = promptarr[0].replace("\n", ",").replace("*","")

                                pattern = r"[^a-zA-Z'\s]"
                                prompt = re.sub(pattern, "", prompt)
                                prompt = prompt.replace("  ", ",")
                            except:
                                prompt = evt.content().replace("\n", "")

                        else:
                            prompt = ""
                elif tag.as_vec()[0] == 'param':
                    if tag.as_vec()[1] == "prompt" or tag.as_vec()[1] == "extra_prompt":
                        extra_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "negative_prompt":
                        negative_prompt = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "lora":
                        lora = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "ratio":
                        if len(tag.as_vec()) > 3:
                            ratio_width = (tag.as_vec()[2])
                            ratio_height = (tag.as_vec()[3])
                        elif len(tag.as_vec()) == 3:
                            split = tag.as_vec()[2].split(":")
                            ratio_width = split[0]
                            ratio_height = split[1]
                    elif tag.as_vec()[1] == "upscale":
                        upscale = tag.as_vec()[2]
                    elif tag.as_vec()[1] == "model":
                        model = tag.as_vec()[2]

            prompt = prompt.replace(";",",")
            request_form["optStr"] = ('prompt=' + prompt + ';extra_prompt=' + extra_prompt + ';negative_prompt='
                                      + negative_prompt + ';upscale=' + str(upscale) + ';model=' + model
                                      + ';ratiow=' + str(ratio_width) + ';ratioh=' + str(ratio_height)) + ';lora=' + str(lora)


        elif task == "image-reimagine":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = task
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    input_type = tag.as_vec()[2]
                    if input_type == "url":
                        url = tag.as_vec()[1]
                    elif input_type == "job":
                        evt = get_referenced_event_by_id(tag.as_vec()[1], [65001], client, config=dvmconfig)
                        if evt is not None:
                            url = evt.content()


            request_form["optStr"] = 'url=' + url



        elif task == "image-upscale":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = task
            upscale = "4"
            url = ""
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    input_type = tag.as_vec()[2]
                    if input_type == "url":
                        url = tag.as_vec()[1]
                        print(url)
                    elif input_type == "event":
                        evt = get_event_by_id(tag.as_vec()[1], config=dvmconfig)
                        url = re.search("(?P<url>https?://[^\s]+)", evt.content()).group("url")
                    elif input_type == "job":
                        evt = get_referenced_event_by_id(tag.as_vec()[1], [65001], client, config=dvmconfig)
                        url = evt.content()
                elif tag.as_vec()[0] == 'param':
                    if tag.as_vec()[1] == "upscale":
                        upscale = tag.as_vec()[2]

            request_form["optStr"] = 'url=' + url + ";upscale=" + upscale

        elif task == "chat":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = task
            text = ""
            user = event.pubkey().to_hex()
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    text = tag.as_vec()[1].replace(";", "")
                elif tag.as_vec()[0] == 'p':
                    user = tag.as_vec()[1]
            request_form["optStr"] = 'message=' + text + ';user=' + user


        elif task == "summarization":
            pattern = r"[^a-zA-Z0-9.!?'\s]"
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = task
            text = ""
            all_texts = ""
            user = event.pubkey().to_hex()
            for tag in event.tags():
                if tag.as_vec()[0] == 'i':
                    input_type = tag.as_vec()[2]
                    if input_type == "text":
                        text = re.sub(pattern, "", tag.as_vec()[1]).replace("\n", "")
                    elif input_type == "event":
                        evt = get_event_by_id(tag.as_vec()[1], config=dvmconfig)
                        text = re.sub(pattern, "", evt.content()).replace("\n", "")
                    elif input_type == "job":
                        evt = get_referenced_event_by_id(tag.as_vec()[1], [65001], client, config=dvmconfig)
                        if evt is not None:
                            text = evt.content().replace(";", "")
                    all_texts = all_texts + text
                elif tag.as_vec()[0] == 'p':
                    user = tag.as_vec()[1]
                request_form["optStr"] = 'user=' + user + ';system_prompt=' + "return a summarization of the given input, no smalltalk. input might contain mutliple articles, separated by three new lines" + ';message=' + all_texts.replace('\U0001f919', "").replace("=", "equals").replace(";", ",")


        elif task == "inactive-following":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = task
            days = "30"
            number = "10"
            user = event.pubkey().to_hex()
            for tag in event.tags():
                if tag.as_vec()[0] == 'param':
                    if tag.as_vec()[1] == 'since':
                        days = tag.as_vec()[2]
                    elif tag.as_vec()[1] == 'user':
                        user = tag.as_vec()[2]
                        if user.startswith("npub"):
                            user = PublicKey.from_bech32(user).to_hex()
            request_form["optStr"] = 'user=' + user + ';since=' + days + ';is_bot=' + str(is_bot)

        elif task == "note-recommendation":
            request_form["mode"] = "PREDICT_STATIC"
            request_form["trainerFilePath"] = task
            days = "1"
            user = event.pubkey().to_hex()
            for tag in event.tags():
                if tag.as_vec()[0] == 'param':
                    if tag.as_vec()[1] == 'since':
                        days = tag.as_vec()[2]
                    elif tag.as_vec()[1] == 'user':
                         user = tag.as_vec()[2]
                         if user.startswith("npub"):
                                user = PublicKey.from_bech32(user).to_hex()


            request_form["optStr"] = 'user=' + user + ';since=' + days + ';is_bot=' + str(is_bot)

        return request_form







    def do_work(job_event, is_from_bot=False):
        if (65002 <= job_event.kind() <= 66000) or job_event.kind() == 4 or job_event.kind() == 68001:
            request_form = create_requestform_from_nostr_event(job_event, is_from_bot)
            task = get_task(job_event, client=client)
            if task == "speech-to-text":
                print("[Nostr] Adding Nostr speech-to-text Job event: " + job_event.as_json())
                input_value = ""
                input_type = ""
                for tag in job_event.tags():
                    if tag.as_vec()[0] == 'i':
                        input_value = tag.as_vec()[1]
                        input_type = tag.as_vec()[2]
                        break
                print("Organizing input..")
                success, duration = organize_input_data(input_value, input_type, request_form, config=dvmconfig)
                print("Organizing input..done.")
                if success is None:
                    respond_to_error("Error processing video", job_event.as_json(), is_from_bot)
                    return
            elif task.startswith("unknown"):
                print("Task not (yet) supported")
                return
            else:
                print("[Nostr] Scheduling " + task + " Job event: " + job_event.as_json())

            url = 'http://' + os.environ["NOVA_HOST"] + ':' + os.environ["NOVA_PORT"] + '/' + str(
                request_form["mode"]).lower()
            headers = {'Content-type': 'application/x-www-form-urlencoded'}
            print("Sending job to NOVA-Server")
            requests.post(url, headers=headers, data=request_form)

    client.handle_notifications(NotificationHandler())

    while True:
        if not dvmconfig.IS_BOT:
            for job in job_list:
                if job.bolt11 != "" and job.payment_hash != "" and not job.is_paid:
                    if str(check_bolt11_ln_bits_is_paid(job.payment_hash)) == "True":
                        job.is_paid = True
                        event = get_event_by_id(job.event_id, config=dvmconfig)
                        if event != None:
                            send_job_status_reaction(event, "processing", True, 0, client=client,  config=dvmconfig)
                            #job_list.remove(job)
                            print("do work from joblist")
                            #job_list.remove(job)
                            do_work(event, is_from_bot=False)
                    elif check_bolt11_ln_bits_is_paid(job.payment_hash) is None: #invoice expired
                        job_list.remove(job)
                        #event = get_event_by_id(job.event_id)
                        #send_job_status_reaction(event, "invoice-expired", False, 0, client=client)

                if Timestamp.now().as_secs() > job.expires:
                    job_list.remove(job)

            for job in jobs_on_hold_list:
                if check_event_has_not_unifinished_job_input(job.event, False, client=client, dvmconfig=dvmconfig):
                    handle_nip90_job_event(job.event)
                    jobs_on_hold_list.remove(job)

                if Timestamp.now().as_secs() > job.timestamp + 60*20: #remove jobs to look for after 20 minutes..
                    jobs_on_hold_list.remove(job)


        #if len(job_list) > 0:
        #    print(str(job_list))

        time.sleep(5.0)


# SEND AND RECEIVE EVENTS
def get_event_by_id(event_id, client=None, config=None):
    dvmconfig = config
    is_new_client = False
    if client is None:
        keys = Keys.from_sk_str(dvmconfig.PRIVATE_KEY)
        client = Client(keys)
        for relay in dvmconfig.RELAY_LIST:
            client.add_relay(relay)
        client.connect()
        is_new_client = True

    split = event_id.split(":")
    if len(split) == 3:
        id_filter = Filter().author(split[1]).custom_tag(Alphabet.D, [split[2]])
        events = client.get_events_of([id_filter], timedelta(seconds=dvmconfig.RELAY_TIMEOUT))
    else:
        id_filter = Filter().id(event_id).limit(1)
        events = client.get_events_of([id_filter], timedelta(seconds=dvmconfig.RELAY_TIMEOUT))
    if is_new_client:
        client.disconnect()
    if len(events) > 0:
        return events[0]
    else:
        return None

def get_referenced_event_by_id(event_id, kinds=None, client=None, config=None):
    #return the event this event is referred in
    dvmconfig = config
    if kinds is None:
        kinds = []
    is_new_client = False
    if client is None:
        keys = Keys.from_sk_str(dvmconfig.PRIVATE_KEY)
        client = Client(keys)
        for relay in dvmconfig.RELAY_LIST:
            client.add_relay(relay)
        client.connect()
        is_new_client = True
    if kinds is None:
        kinds = []
    if len(kinds) > 0:
        job_id_filter = Filter().kinds(kinds).event(EventId.from_hex(event_id)).limit(1)
    else:
        job_id_filter = Filter().event(EventId.from_hex(event_id)).limit(1)



    events = client.get_events_of([job_id_filter], timedelta(seconds=dvmconfig.RELAY_TIMEOUT))

    if is_new_client:
        client.disconnect()
    if len(events) > 0:
        return events[0]
    else:
        return None

def send_event(event, client=None, key=None):
    relays = []
    is_new_client = False

    for tag in event.tags():
        if tag.as_vec()[0] == 'relays':
            relays = tag.as_vec()[1].split(',')

    if client is None:

        if key is None:
            key = Keys.from_sk_str(dvmconfig.PRIVATE_KEY)
        opts = Options().wait_for_ok(False)
        client = Client.with_opts(key, opts)
        for relay in dvmconfig.RELAY_LIST:
            client.add_relay(relay)
        client.connect()
        is_new_client = True

    for relay in relays:
        if relay not in dvmconfig.RELAY_LIST:
            client.add_relay(relay)
    client.connect()

    event_id = client.send_event(event)
    #event_id = client.send_msg(ClientMessage.EV(event))

    for relay in relays:
        if relay not in dvmconfig.RELAY_LIST:
            client.remove_relay(relay)

    if is_new_client:
        client.disconnect()

    return event_id


# GET INFO ON TASK
def get_task(event, client):
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
                elif tag.as_vec()[2] == "event":
                    evt = get_event_by_id(tag.as_vec()[1], config=dvmconfig)
                    if evt is not None:
                        if evt.kind() == 1063:
                            for tag in evt.tags():
                                if tag.as_vec()[0] == 'url':

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
        has_image_tag = False
        has_text_tag = False
        for tag in event.tags():
            if tag.as_vec()[0] == "i":
                if tag.as_vec()[2] == "url":
                    file_type = check_url_is_readable(tag.as_vec()[1])
                    if file_type == "image":
                        has_image_tag = True
                        print("found image tag")
                elif tag.as_vec()[2] == "job":
                    evt = get_referenced_event_by_id(tag.as_vec()[1], [65001], client, config=dvmconfig)
                    if evt is not None:
                        file_type = check_url_is_readable(evt.content())
                        if file_type == "image":
                            has_image_tag = True
                elif tag.as_vec()[2] == "text":
                    has_text_tag = True


        if has_image_tag and not has_text_tag:
            return "image-reimagine"
        elif has_image_tag and has_text_tag:
            return "image-to-image"
        elif has_text_tag and not has_image_tag:
            return "text-to-image"
    elif event.kind() == 65006:
        return "note-recommendation"
    elif event.kind() == 65007:
        return "inactive-following"
    else:
        return "unknown type"

def check_task_is_supported(event, client, get_duration = False, config=None):
    dvmconfig = config
    input_value = ""
    input_type = ""
    duration = 1
    start = "0"
    end = "0"
    output_is_set = True

    for tag in event.tags():
        if tag.as_vec()[0] == 'i':
            if len(tag.as_vec()) < 3:
                print("Job Event missing/malformed i tag, skipping..")
                return False, "", 0
            else:
                input_value = tag.as_vec()[1]
                input_type = tag.as_vec()[2]
                if input_type == "event":
                   evt = get_event_by_id(input_value, config=dvmconfig)
                   if evt == None:
                       print("Event not found")
                       return False, "", 0

        elif tag.as_vec()[0] == 'output':
                output = tag.as_vec()[1]
                output_is_set = True
                if not (output == "text/plain" or output == "text/json" or output == "json" or output == "image/png" or "image/jpg" or output == ""):
                    print("Output format not supported, skipping..")
                    return False, "", 0
                else:
                    print("Output Format: " + output)


        elif tag.as_vec()[0] == 'param':
            if get_duration:
                param = tag.as_vec()[1]
                if param == "range":  # check for paramtype
                    try:
                        t = time.strptime(tag.as_vec()[2], "%H:%M:%S")
                        seconds = t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec
                        start = str(seconds)
                    except:
                        try:
                            t = time.strptime(tag.as_vec()[2], "%M:%S")
                            seconds = t.tm_min * 60 + t.tm_sec
                            start = str(seconds)
                        except:
                            start = tag.as_vec()[2]
                    try:
                        t = time.strptime(tag.as_vec()[3], "%H:%M:%S")
                        seconds = t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec
                        end = str(seconds)
                    except:
                        try:
                            t = time.strptime(tag.as_vec()[3], "%M:%S")
                            seconds = t.tm_min * 60 + t.tm_sec
                            end = str(seconds)
                        except:
                            end = tag.as_vec()[3]

    task = get_task(event, client=client)
    if not output_is_set:
        print("No output set")
    if task not in dvmconfig.SUPPORTED_TASKS:  # The Tasks this DVM supports (can be extended)
        return False, task, duration
    elif task == "translation" and (
            input_type != "event" and input_type != "job" and input_type != "text"):  # The input types per task
        return False, task, duration
    if task == "translation" and input_type != "text" and len(event.content()) > 4999:  # Google Services have a limit of 5000 signs
        return False, task, duration
    elif task == "summarization" and (
            input_type != "event" and input_type != "job" and input_type != "text"):  # The input types per task
        return False, task, duration

    elif task == "speech-to-text" and (
            input_type != "event" and input_type != "job" and input_type != "url"):  # The input types per task
            print("nope..")
            return False, task, duration



    elif task == "image-upscale" and (input_type != "event" and input_type != "job" and input_type != "url"):
        return False, task, duration
    if input_type == 'url' and check_url_is_readable(input_value) is None:
        print("url not readable")
        return False, task, duration

    if task == 'speech-to-text' and get_duration:
        print("getting duration..")
        duration = check_media_length(input_value, input_type, event.id().to_hex(), start, end)
        return True, task, duration


    return True, task, duration

def get_amount_per_task(task, duration = 0, config=None):
    dvmconfig = config
    if task == "translation":
        amount = dvmconfig.COSTPERUNIT_TRANSLATION
    elif task == "speech-to-text":
        if duration == 0:
            amount = dvmconfig.COSTPERUNIT_SPEECHTOTEXT * 1000 #if we dont have a duration make it rate x 1000.
        else:
            amount = 50 + int(dvmconfig.COSTPERUNIT_SPEECHTOTEXT * duration)
    elif task == "text-to-image":
        amount = dvmconfig.COSTPERUNIT_IMAGEGENERATION
    elif task == "image-to-image":
        amount = dvmconfig.COSTPERUNIT_IMAGETRANSFORMING
    elif task == "image-reimagine":
        amount = dvmconfig.COSTPERUNIT_IMAGETRANSFORMING
    elif task == "image-upscale":
        amount = dvmconfig.COSTPERUNIT_IMAGEUPSCALING
    elif task == "chat":
        amount = 0
    elif task == "image-to-text":
        amount = dvmconfig.COSTPERUNIT_IMAGEGENERATION
    elif task == "summarization":
        amount = dvmconfig.COSTPERUNIT_IMAGEGENERATION
    elif task == "inactive-following":
        amount = dvmconfig.COSTPERUNIT_INACTIVE_FOLLOWING
    elif task == "note-recommendation":
        amount = dvmconfig.COSTPERUNIT_NOTE_RECOMMENDATION
    else:
        print("[Nostr] Task " + task + " is currently not supported by this instance, skipping")
        return None
    return amount


# DECIDE TO RETURN RESULT
def check_event_status(data, original_event_str: str, dvm_key="", use_bot=False):
    original_event = Event.from_json(original_event_str)
    keys = Keys.from_sk_str(dvm_key)

    for x in job_list:
        if x.event_id == original_event.id().to_hex():
            is_paid = x.is_paid
            amount = x.amount
            x.result = data
            x.is_processed = True
            if dvmconfig.SHOWRESULTBEFOREPAYMENT and not is_paid:
                send_nostr_reply_event(data, original_event_str, key=keys)
                send_job_status_reaction(original_event, "success", amount, config=dvmconfig)  # or payment-required, or both?
            elif not dvmconfig.SHOWRESULTBEFOREPAYMENT and not is_paid:
                send_job_status_reaction(original_event, "success", amount, config=dvmconfig)  # or payment-required, or both?

            if dvmconfig.SHOWRESULTBEFOREPAYMENT and is_paid:
                job_list.remove(x)
            elif not dvmconfig.SHOWRESULTBEFOREPAYMENT and is_paid:
                job_list.remove(x)
                send_nostr_reply_event(data, original_event_str, key=keys)
            break

    post_processed_content = post_process_result(data, original_event)
    print(str(job_list))


    if use_bot:
        receiver_key = PublicKey()
        for tag in original_event.tags():
            if tag.as_vec()[0] == "p": #TODO maybe use another tag, e.g y, as p might be overwritten in some events.
                receiver_key = PublicKey.from_hex(tag.as_vec()[1])
        event = EventBuilder.new_encrypted_direct_msg(keys, receiver_key, post_processed_content, None).to_event(keys)
        send_event(event, key=keys)

    else:
        send_nostr_reply_event(post_processed_content, original_event_str, key=keys)
        #send_job_status_reaction(original_event, "success")


# NIP90 REPLIES
def respond_to_error(content, originaleventstr, is_from_bot=False, dvm_key=None ):
    if dvm_key is None:
        keys = Keys.from_sk_str(dvmconfig.PRIVATE_KEY)
    else:
        keys = Keys.from_sk_str(dvm_key)

    original_event = Event.from_json(originaleventstr)
    sender = ""
    task = ""
    if not is_from_bot:
        send_job_status_reaction(original_event, "error", content=content, key=dvm_key)
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
            amount = int(user[1]) + get_amount_per_task(task)
            update_sql_table(sender, amount, user[2], user[3], user[4], user[5], user[6],
                             Timestamp.now().as_secs())
            message = "There was the following error : " + content + ". Credits have been reimbursed"
        else:
            # User didn't pay, so no reimbursement
            message = "There was the following error : " + content

        evt = EventBuilder.new_encrypted_direct_msg(keys, PublicKey.from_hex(sender), message, None).to_event(keys)
        send_event(evt, key=keys)


def send_nostr_reply_event(content, original_event_as_str, key=None):
    originalevent = Event.from_json(original_event_as_str)
    requesttag = Tag.parse(["request", original_event_as_str.replace("\\", "")])
    etag = Tag.parse(["e", originalevent.id().to_hex()])
    ptag = Tag.parse(["p", originalevent.pubkey().to_hex()])
    alttag = Tag.parse(["alt", "This is the result of a NIP90 DVM AI task with kind " + str(
        originalevent.kind()) + ". The task was: " + originalevent.content()])
    statustag = Tag.parse(["status", "success"])

    if key is None:
        key = Keys.from_sk_str(dvmconfig.PRIVATE_KEY)

    event = EventBuilder(65001, str(content), [requesttag, etag, ptag, alttag, statustag]).to_event(key)
    send_event(event, key=key)
    print("[Nostr] 65001 Job Response event sent: " + event.as_json())
    return event.as_json()


def send_job_status_reaction(original_event, status, is_paid=True, amount=0, client=None, content=None, config=None, key=None):
    dvmconfig = config
    altdesc = "This is a reaction to a NIP90 DVM AI task. "
    task = get_task(original_event, client=client)
    if status == "processing":
        altdesc = "NIP90 DVM AI task " + task + " started processing. "
        reaction = altdesc + emoji.emojize(":thumbs_up:")
    elif status == "success":
        altdesc = "NIP90 DVM AI task " + task + " finished successfully. "
        reaction = altdesc + emoji.emojize(":call_me_hand:")
    elif status == "chain-scheduled":
        altdesc = "NIP90 DVM AI task " + task + " Chain Task scheduled"
        reaction = altdesc + emoji.emojize(":thumbs_up:")
    elif status == "error":
        altdesc = "NIP90 DVM AI task " + task + " had an error. "
        if content is None:
            reaction = altdesc + emoji.emojize(":thumbs_down:")
        else:
            reaction = altdesc + emoji.emojize(":thumbs_down:") + content

    elif status == "payment-required":

        altdesc = "NIP90 DVM AI task " + task + " requires payment of min " + str(amount) + " Sats. "
        # if task == "speech-to-text":
            #altdesc = altdesc + (" Providing results with WhisperX. "
            #                     "Accepted input formats: wav,mp3,mp4,ogg,avi,mov,youtube,overcast. "
             #                    "Possible outputs: text/plain, timestamped labels depending on "
             #                    "alignment parameter (word,segment,raw) ")
        #elif task == "image-to-text":
            #altdesc = altdesc + (" Accepted input formats: jpg. Possible outputs: text/plain. "
             #                    "This is very experimental, make sure your text is well readable. ")
        reaction = altdesc + emoji.emojize(":orange_heart:")

    elif status == "payment-rejected":
        altdesc = "NIP90 DVM AI task " + task + " payment is below required amount of " + str(amount) + " Sats. "
        reaction = altdesc + emoji.emojize(":thumbs_down:")
    elif status == "user-blocked-from-service":

        altdesc = "NIP90 DVM AI task " + task + " can't be performed. User has been blocked from Service. "
        reaction = altdesc + emoji.emojize(":thumbs_down:")
    else:
        reaction = emoji.emojize(":thumbs_down:")

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
    payment_hash = ""
    expires = original_event.created_at().as_secs() + (60*60*24)
    if status == "payment-required" or (status == "processing" and not is_paid):
        if dvmconfig.LNBITS_INVOICE_KEY != "":
            try:
                bolt11, payment_hash = create_bolt11_ln_bits(amount)
            except Exception as e:
                print(e)



    if not any(x.event_id == original_event.id().to_hex() for x in job_list):
        job_list.append(
            JobToWatch(event_id=original_event.id().to_hex(), timestamp=original_event.created_at().as_secs(), amount=amount,
                       is_paid=is_paid,
                       status=status, result="", is_processed=False, bolt11=bolt11, payment_hash=payment_hash, expires=expires, from_bot=False))
        print(str(job_list))
    if status == "payment-required" or status == "payment-rejected" or (status == "processing" and not is_paid) or (
            status == "success" and not is_paid):

        if dvmconfig.LNBITS_INVOICE_KEY != "":
            amount_tag = Tag.parse(["amount", str(amount * 1000), bolt11])
        else:
            amount_tag = Tag.parse(["amount", str(amount * 1000)])  # to millisats
        tags.append(amount_tag)
    if key is not None:
        keys = Keys.from_sk_str(key)
    else:
        keys = Keys.from_sk_str(dvmconfig.PRIVATE_KEY)
    event = EventBuilder(65000, reaction, tags).to_event(keys)

    send_event(event, key=keys)
    print("[Nostr] Sent Kind 65000 Reaction: " + status + " " + event.as_json())
    return event.as_json()


# POSTPROCESSING
def post_process_result(anno, original_event):
    print("post-processing...")
    if isinstance(anno, Anno): #if input is an anno we parse it to required output format
        for tag in original_event.tags():
            print(tag.as_vec()[0])
            if tag.as_vec()[0] == "output":
                print("HAS OUTPUT TAG")
                output_format = tag.as_vec()[1]
                print("requested output is " + str(tag.as_vec()[1]) + "...")
                try:
                    if output_format == "text/plain":
                        result = ""
                        print(str(anno.data))
                        for element in anno.data:
                            name = element["name"] #name
                            cleared_name = str(name).lstrip("\'").rstrip("\'")
                            result = result + cleared_name + "\n"
                        result = replace_broken_words(str(result).replace("\"", "").replace('[', "").replace(']', "").lstrip(None))
                        return result

                    elif output_format == "text/json" or output_format == "json":
                        #result = json.dumps(json.loads(anno.data.to_json(orient="records")))
                        result =  replace_broken_words(json.dumps(anno.data.tolist()))
                        return result
                    # TODO add more
                    else:
                        result = ""
                        for element in anno.data:
                            element["name"] = str(element["name"]).lstrip()
                            element["from"] = (format(float(element["from"]), '.2f')).lstrip()  # name
                            element["to"] = (format(float(element["to"]), '.2f')).lstrip()  # name
                            result = result + "(" + str(element["from"]) + "," + str(element["to"]) + ")" + " " + str(
                                element["name"]) + "\n"

                        print(result)
                        result = replace_broken_words(result)
                        return result

                except Exception as e:
                    print(e)
                    result =  replace_broken_words(str(anno.data))
                    return result

        else:
            result = ""
            for element in anno.data:
                element["name"] = str(element["name"]).lstrip()
                element["from"] = (format(float(element["from"]), '.2f')).lstrip()  # name
                element["to"] = (format(float(element["to"]), '.2f')).lstrip()  # name
                result = result + "(" + str(element["from"]) + "," +  str(element["to"]) +")" + " " + str(element["name"]) + "\n"

            print(result)
            result = replace_broken_words(result)
            return result
    elif isinstance(anno, NoneType):
        return "An error occured"
    else:
        result = replace_broken_words(anno)
        return result


def replace_broken_words(text):
    result = (text.replace("Noster", "Nostr").replace("Nostra", "Nostr").replace("no stir", "Nostr").
              replace("Nostro", "Nostr").replace("Impub", "npub").replace("sets", "Sats"))
    return result

# BOT FUNCTIONS
def get_bot_help_text():
    return (
            "Hi there. I'm a bot interface to the first NIP90 Data Vending Machine and I can perform several AI tasks"
            " for you. Currently I can do the following jobs:\n\n"
            "Generate an Image with Stable Diffusion XL (" + str(dvmconfig.COSTPERUNIT_IMAGEGENERATION) + " Sats)\n"
            "-text-to-image someprompt\nAdditional parameters:\n-negative some negative prompt\n-ratio width:height "
            "(e.g. 3:4), default 1:1\n-lora specific weights (only XL models): "
            "\"3d_render_style_xl\", \"cyborg_style_xl\", \"psychedelic_noir_xl\", \"dreamarts_xl\", \"voxel_xl\", \"kru3ger_xl\", \"wojak_xl\", \"ink_punk_xl\"\n"
            "-model anothermodel\nOther Models are: \"dreamshaper\",\"nightvision\",\"protovision\",\"dynavision\",\"sdvn\",\"fantastic\",\"chroma\",\"crystalclear\". Non-XL models are: \"wild\",\"realistic\",\"lora_inks\",\"lora_pepe\"\n\n"
            "Transform an existing Image with Stable Diffusion XL (" + str(dvmconfig.COSTPERUNIT_IMAGETRANSFORMING)
            + " Sats)\n" "-image-to-image urltoimagedotjpg in style of an oil painting by pablo picasso\n Alternative model is \"pix2pix\". E.g: -image-to-image urltoimagedotjpg turn him into a ghost -model pix2pix\n\n"
            "Parse text from an Image (make sure text is well readable) (" + str(dvmconfig.COSTPERUNIT_OCR) + " Sats)\n"
            "-image-to-text urltofile \n\n"
            "Upscale the resolution of an Image 4x and improve quality (" + str(dvmconfig.COSTPERUNIT_IMAGEUPSCALING)
            + " Sats)\n -image-upscale urltofile \n\n"
            "Transcribe Audio/Video/Youtube/Overcast from an URL with WhisperX (" +
            str(get_amount_per_task("speech-to-text", config=dvmconfig)) + " Sats)\n"
            "-speech-to-text urltofile \nAdditional parameters:\n-from timeinseconds -to timeinseconds\n\n"
            "Get a List of inactive users you follow (" + str(dvmconfig.COSTPERUNIT_INACTIVE_FOLLOWING) + " Sats)\n"
            "-inactive-following\nAdditional parameters:\n-sincedays days (e.g. 60), default 30\n\n"
            "To show your current balance\n -balance \n\n"
            "You can zap any of my notes/dms or my profile to top up your balance. I also understand Zapplepay.")


def parse_bot_command_to_event(dec_text, sender):
    dec_text = dec_text.replace("\n", " ")
    if str(dec_text).startswith("-text-to-image"):
        command = dec_text.replace("-text-to-image ", "")
        split = command.split(" -")
        prompt = split[0]
        ratiow = "1"
        ratioh = "1"
        lora = ""
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
                elif i.startswith("lora "):
                    lora = i.replace("lora ", "")
                    param_tag = Tag.parse(["param", "lora", lora])
                    tags.append(param_tag)
                elif i.startswith("model "):
                    model = i.replace("model ", "")
                    param_tag = Tag.parse(["param", "model", model])
                    tags.append(param_tag)
                elif i.startswith("ratio "):
                    ratio = str(i.replace("ratio ", ""))
                    split = ratio.split(":")
                    ratiow = split[0]
                    ratioh = split[1]

        param_ratio_tag = Tag.parse(["param", "ratio", ratiow, ratioh])
        tags.append(param_ratio_tag)

        return tags

    elif str(dec_text).startswith("-image-to-image"):
        dec_text = dec_text.replace("\n", " ")
        prompt = ""

        command = dec_text.replace("-image-to-image ", "")
        split = command.split(" -")
        urltemp = str(split[0]).split(" ")
        url = urltemp[0]
        j_tag = Tag.parse(["j", "image-to-image"])
        i_tag = Tag.parse(["i", url, "url"])
        tags = [j_tag, i_tag]

        if len(urltemp) > 1:
            for j in range(1,len(urltemp)):
                prompt = prompt + urltemp[j] + " "
            i_tag_2 = Tag.parse(["i", prompt.rstrip(), "text"])
            tags.append(i_tag_2)


        if len(split) > 1:
            for i in split:
                if i.startswith("negative "):
                    negative_prompt = i.replace("negative ", "")
                    param_tag = Tag.parse(["param", "negative_prompt", negative_prompt])
                    tags.append(param_tag)
                elif i.startswith("prompt "):
                    prompt = i.replace("prompt ", "")
                    i_tag_2 = Tag.parse(["i", prompt, "text"])
                    tags.append(i_tag_2)
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
                elif i.startswith("lora "):
                    lora = i.replace("lora ", "")
                    param_tag = Tag.parse(["param", "lora", lora])
                    tags.append(param_tag)

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

    elif str(dec_text).startswith("-image-reimagine"):
        command = dec_text.replace("-image-reimagine ", "")
        split = command.split(" -")
        url = str(split[0]).replace(' ', '')
        j_tag = Tag.parse(["j", "image-reimagine"])
        i_tag = Tag.parse(["i", url, "url"])
        tags = [j_tag, i_tag]

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
        model = "base" #"large-v2"
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
        user = sender
        command = dec_text.replace("-inactive-following", "")
        split = command.split(" -")
        for i in split:
            if i.startswith("sincedays "):
                since_days = i.replace("sincedays ", "")
                print("Since days: " + str(since_days))
            elif i.startswith("user "):
                user = i.replace("user ", "")
                print("User: " + str(user))

        param_tag_since = Tag.parse(["param", "since", since_days])
        param_tag_user = Tag.parse(["param", "user", user])
        j_tag = Tag.parse(["j", "inactive-following"])
        return [j_tag, param_tag_since, param_tag_user]

    elif str(dec_text).startswith("-note-recommendation"):
        since_days = "1"
        user = sender
        command = dec_text.replace("-note-recommendation", "")
        split = command.split(" -")
        for i in split:
            if i.startswith("sincedays "):
                since_days = i.replace("sincedays ", "")
                print("Since days: " + str(since_days))
            elif i.startswith("user "):
                user = i.replace("user ", "")
                print("User: " + str(user))

        param_tag_since = Tag.parse(["param", "since", since_days])
        param_tag_user = Tag.parse(["param", "user", user])
        j_tag = Tag.parse(["j", "note-recommendation"])
        return [j_tag, param_tag_since, param_tag_user]
    else:
        text = dec_text
        j_tag = Tag.parse(["j", "chat"])
        i_tag = Tag.parse(["i", text, "text"])
        return [j_tag, i_tag]


def check_event_has_not_unifinished_job_input(nevent, append, client, dvmconfig):

    tasksupported, task, duration = check_task_is_supported(nevent, client, False, config=dvmconfig)
    if not tasksupported:
        return False

    for tag in nevent.tags():
        if tag.as_vec()[0] == 'i':
            if len(tag.as_vec()) < 3:
                print("Job Event missing/malformed i tag, skipping..")
                return False
            else:
                input = tag.as_vec()[1]
                input_type = tag.as_vec()[2]
                if input_type == "job":
                    evt = get_referenced_event_by_id(input, [65001], client, config=dvmconfig)
                    if evt is None:
                        if append:
                            job = RequiredJobToWatch(event=nevent, timestamp=Timestamp.now().as_secs())
                            jobs_on_hold_list.append(job)
                            send_job_status_reaction(nevent, "chain-scheduled", True, 0, client=client, config=dvmconfig)

                        return False
    else:
        return True


# CHECK INPUTS/TASK AVAILABLE

    input_value = ""
    input_type = ""
    duration = 1
    start = "0"
    end = "0"

    for tag in event.tags():
        if tag.as_vec()[0] == 'i':
            if len(tag.as_vec()) < 3:
                print("Job Event missing/malformed i tag, skipping..")
                return False, "", 0
            else:
                input_value = tag.as_vec()[1]
                input_type = tag.as_vec()[2]
                if input_type == "event":
                   evt = get_event_by_id(input_value, config=dvmconfig)
                   if evt == None:
                       print("Event not found")
                       return False, "", 0

        elif tag.as_vec()[0] == 'output':
                output = tag.as_vec()[1]
                output_is_set = True
                if not (output == "text/plain" or output == "text/json" or output == "json" or output == "image/png" or "image/jpg" or output == ""):
                    return False, "", 0



        elif tag.as_vec()[0] == 'param':
            if get_duration:
                param = tag.as_vec()[1]
                if param == "range":  # check for paramtype
                    try:
                        t = time.strptime(tag.as_vec()[2], "%H:%M:%S")
                        seconds = t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec
                        start = str(seconds)
                    except:
                        try:
                            t = time.strptime(tag.as_vec()[2], "%M:%S")
                            seconds = t.tm_min * 60 + t.tm_sec
                            start = str(seconds)
                        except:
                            start = tag.as_vec()[2]
                    try:
                        t = time.strptime(tag.as_vec()[3], "%H:%M:%S")
                        seconds = t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec
                        end = str(seconds)
                    except:
                        try:
                            t = time.strptime(tag.as_vec()[3], "%M:%S")
                            seconds = t.tm_min * 60 + t.tm_sec
                            end = str(seconds)
                        except:
                            end = tag.as_vec()[3]

    task = get_task(event, client=client)
    if task not in dvmconfig.SUPPORTED_TASKS:  # The Tasks this DVM supports (can be extended)
        print("Not in supported tasks")
        return False, task, duration
    elif task == "translation" and (
            input_type != "event" and input_type != "job" and input_type != "text"):  # The input types per task
        return False, task, duration
    if task == "translation" and input_type != "text" and len(event.content()) > 4999:  # Google Services have a limit of 5000 signs
        return False, task, duration
    elif task == "summarization" and (
            input_type != "event" and input_type != "job" and input_type != "text"):  # The input types per task
        return False, task, duration

    elif task == "speech-to-text" and (
            input_type != "event" and input_type != "job" and input_type != "url"):  # The input types per task
            return False, task, duration



    elif task == "image-upscale" and (input_type != "event" and input_type != "job" and input_type != "url"):
        return False, task, duration

   # elif task == "image-reimagine" and input_type != "url":
   #     return False, task, duration
    if input_type == 'url' and check_url_is_readable(input_value) is None:
        return False, task, duration

    if task == 'speech-to-text' and get_duration:
        duration = check_media_length(input_value, input_type, event.id().to_hex(), start, end)
        return True, task, duration


    return True, task, duration


def get_overcast(input_value, request_form):
    filename = os.environ["NOVA_DATA_DIR"] + '\\' + request_form["database"] + '\\' + request_form["sessions"] + '\\' + \
               request_form[
                   "roles"] + ".originalaudio.mp3"
    print("Found overcast.fm Link.. downloading")
    start = request_form["startTime"]
    end = request_form["endTime"]
    download_podcast(input_value, filename)
    finaltag = str(input_value).replace("https://overcast.fm/", "").split('/')
    if int(request_form["startTime"]) == 0:
        if len(finaltag) > 1:
            t = time.strptime(finaltag[1], "%H:%M:%S")
            seconds = t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec
            start = str(seconds)  # overwrite from link.. why not..
            print("Setting start time automatically to " + start)
            if float(request_form["endTime"]) > 0.0:
                end = seconds + float(request_form["endTime"])
                print("Moving end time automatically to " + end)

    return filename, start, end


def get_youtube(input_value, request_form):
    filepath = os.environ["NOVA_DATA_DIR"] + '\\' + request_form["database"] + '\\' + request_form["sessions"] + '\\'
    start = request_form["startTime"]
    end = request_form["endTime"]
    try:
        filename = downloadYouTube(input_value, filepath)

    except Exception as e:
        print(e)
        return filename, start, end
    try:
        o = urlparse(input_value)
        q = urllib.parse.parse_qs(o.query)
        if int(request_form["startTime"]) == 0:
            if o.query.find('?t=') != -1:
                start = q['t'][0]  # overwrite from link.. why not..
                print("Setting start time automatically to " + start)
                if float(request_form["endTime"]) > 0.0:
                    end = str(float(q['t'][0]) + float(request_form["endTime"]))
                    print("Moving end time automatically to " + end)

    except Exception as e:
        print(e)
        return filename, start, end

    return filename, start, end


def get_media_link(input_value, request_form):
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
    filename = os.environ["NOVA_DATA_DIR"] + '\\' + request_form["database"] + '\\' + request_form[
        "sessions"] + '\\' + request_form["roles"] + '.original' + file_type + '.' + ext
    print(filename)

    try:
        if not os.path.exists(filename):
            file = open(filename, 'wb')
            for chunk in req.iter_content(100000):
                file.write(chunk)
            file.close()
    except Exception as e:
        print(e)

    return filename


def organize_input_data(input_value, input_type, request_form, process=True, config=None):
    filename = ""
    dvmconfig = config


    if input_type == "event":  # NIP94 event
        evt = get_event_by_id(input_value, config=dvmconfig)
        if evt is not None:
            if evt.kind() == 1063:
                for tag in evt.tags():
                    if tag.as_vec()[0] == 'url':
                        input_type = "url"
                        input_value = tag.as_vec()[1]

    if input_type == "url":
        if not os.path.exists(
                os.environ["NOVA_DATA_DIR"] + '\\' + request_form["database"] + '\\' + request_form["sessions"]):
            os.mkdir(os.environ["NOVA_DATA_DIR"] + '\\' + request_form["database"] + '\\' + request_form["sessions"])

        # We can support some services that don't use default media links, like overcastfm for podcasts
        if str(input_value).startswith("https://overcast.fm/"):
            filename, start, end = get_overcast(input_value, request_form)
            request_form["startTime"] = str(start)
            request_form["endTime"] = str(end)
        # or youtube links..
        elif str(input_value).replace("http://", "").replace("https://", "").replace(
                "www.", "").replace("youtu.be/", "youtube.com?v=")[0:11] == "youtube.com":

            filename, start, end = get_youtube(input_value, request_form)
            request_form["startTime"] = str(start)
            request_form["endTime"] = str(end)

        else:
            filename = get_media_link(input_value, request_form)

        try:

            file_reader = AudioReader(filename, ctx=cpu(0), mono=False)
            duration = file_reader.duration()
        except Exception as e:
            print(e)
            return None, 0

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

        duration = end_time - start_time

    if process:
        print("Converting from " + str(start_time) + " until " + str(end_time))
        # for now we cut and convert all files to mp3
        finalfilename = os.environ["NOVA_DATA_DIR"] + '\\' + request_form["database"] + '\\' + request_form[
            "sessions"] + '\\' + request_form[
                            "roles"] + '.' + request_form["streamName"] + '.mp3'
        fs, x = ffmpegio.audio.read(filename, ss=start_time, to=end_time, sample_fmt='dbl', ac=1)
        ffmpegio.audio.write(finalfilename, fs, x)

        if not db_entry_exists(request_form, request_form["sessions"], "name", "Sessions"):
            add_new_session_to_db(request_form, duration)
    else:
        os.remove(filename)

    return True, duration
def check_media_length(input_value, input_type, id, start, end):
    if input_type == "url":
        request_form = {"database": "nostr_test",
                        "roles": "nostr",
                        "sessions": id,  "startTime": start, "endTime": end}

        print(end + " " + start)
        if float(end) - float(start) > 0.0:
            return int(float(end) - float(start))
        success, duration = organize_input_data(input_value, input_type, request_form, process=False, config=dvmconfig)
        return duration
    elif input_type == "text":
        return len(input_value)
    elif input_type == "job":
        evt = get_referenced_event_by_id(input_value, config=dvmconfig)
        return 1
    elif input_type == "event":
        evt = get_event_by_id(input_value, config=dvmconfig)
        return 1


    return 1

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
            ".jpg") or content_type == 'image/jpeg' or str(url).endswith(".jpeg") or str(url).endswith(".pdf") or content_type == 'image/png' or str(
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
    url = dvmconfig.LNBITS_URL + "/api/v1/payments"
    data = {'out': False, 'amount': sats, 'memo': "Nostr-DVM"}
    headers = {'X-API-Key': dvmconfig.LNBITS_INVOICE_KEY, 'Content-Type': 'application/json', 'charset': 'UTF-8'}
    try:
        res = requests.post(url, json=data, headers=headers)
        obj = json.loads(res.text)
        return obj["payment_request"], obj["payment_hash"]
    except Exception as e:
        print(e)
        return None

def check_bolt11_ln_bits_is_paid(payment_hash):
    url = dvmconfig.LNBITS_URL + "/api/v1/payments/" + payment_hash
    headers = {'X-API-Key': dvmconfig.LNBITS_INVOICE_KEY, 'Content-Type': 'application/json', 'charset': 'UTF-8'}
    try:
        res = requests.get(url, headers=headers)
        obj = json.loads(res.text)
        return obj["paid"]
    except Exception as e:
        #print("Exception checking invoice is paid:" + e)
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
        con = sqlite3.connect(dvmconfig.USERDB)
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
        con = sqlite3.connect(dvmconfig.USERDB)
        cur = con.cursor()
        cur.execute(""" ALTER TABLE users ADD COLUMN lastactive 'integer' """)
        con.close()
    except Error as e:
        print(e)


def add_to_sql_table(npub, sats, iswhitelisted, isblacklisted, nip05, lud16, name, lastactive):
    try:
        con = sqlite3.connect(dvmconfig.USERDB)
        cur = con.cursor()
        data = (npub, sats, iswhitelisted, isblacklisted, nip05, lud16, name, lastactive)
        cur.execute("INSERT or IGNORE INTO users VALUES(?, ?, ?, ?, ?, ?, ?, ?)", data)
        con.commit()
        con.close()
    except Error as e:
        print(e)


def update_sql_table(npub, sats, iswhitelisted, isblacklisted, nip05, lud16, name, lastactive):
    try:
        con = sqlite3.connect(dvmconfig.USERDB)
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
        con = sqlite3.connect(dvmconfig.USERDB)
        cur = con.cursor()
        cur.execute("SELECT * FROM users WHERE npub=?", (npub,))
        row = cur.fetchone()
        con.close()
        return row

    except Error as e:
        print(e)


def delete_from_sql_table(npub):
    try:
        con = sqlite3.connect(dvmconfig.USERDB)
        cur = con.cursor()
        cur.execute("DELETE FROM users WHERE npub=?", (npub,))
        con.commit()
        con.close()
    except Error as e:
        print(e)


def clear_db():
    try:
        con = sqlite3.connect(dvmconfig.USERDB)
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
        con = sqlite3.connect(dvmconfig.USERDB)
        cur = con.cursor()
        cur.execute("SELECT * FROM users ORDER BY sats DESC")
        rows = cur.fetchall()
        for row in rows:
            print(row)
        con.close()
    except Error as e:
        print(e)


def update_user_balance(sender, sats, config=None):
    user = get_from_sql_table(sender)
    if user is None:
        add_to_sql_table(sender, (int(sats) + dvmconfig.NEW_USER_BALANCE), False, False,
                         "", "", "", Timestamp.now().as_secs())
        print("NEW USER: " + sender + " Zap amount: " + str(sats) + " Sats.")
    else:
        user = get_from_sql_table(sender)
        print(str(sats))
        if user[4] is None:
            user[4] = ""
        if user[5] is None:
            user[5] = ""
        if user[6] is None:
            user[6] = ""
        new_balance = int(user[1]) + sats
        update_sql_table(sender, new_balance, user[2], user[3], user[4], user[5], user[6],
                         Timestamp.now().as_secs())
        print("UPDATE USER BALANCE: " + user[6] + " Zap amount: " + str(sats) + " Sats.")


        if config is not None:
            keys = Keys.from_sk_str(config.PRIVATE_KEY)
            time.sleep(1.0)

            message = ("Added "+ str(sats) + " Sats to balance. New balance is " + str(new_balance) + " Sats. " )


            evt = EventBuilder.new_encrypted_direct_msg(keys, PublicKey.from_hex(sender), message,
                                                        None).to_event(keys)
            send_event(evt, key=keys)


def get_or_add_user(sender):
    user = get_from_sql_table(sender)
    if user is None:
        add_to_sql_table(sender, dvmconfig.NEW_USER_BALANCE, False, False, None,
                         None, None, Timestamp.now().as_secs())
        user = get_from_sql_table(sender)
    return user


# ADMINISTRARIVE DB MANAGEMENT
def admin_make_database_updates(config=None):
    # This is called on start of Server, Admin function to manually whitelist/blacklist/add balance/delete users
    # List all entries, why not.
    dvmconfig = config

    rebroadcast_nip89 = False
    cleardb = False
    listdatabase = False
    deleteuser = False
    whitelistuser = False
    unwhitelistuser = True
    blacklistuser = False
    addbalance = False
    additional_balance = 50



    #publickey = PublicKey.from_bech32("npub1at8xw328h5458285f0w6l5wqwsxxfyrj6wsafu5e3hsvyrgtdgysz6zckp").to_hex()
    # use this if you have the npub
    publickey = "99bb5591c9116600f845107d31f9b59e2f7c7e09a1ff802e84f1d43da557ca64"
    publickey = "4564d670cc2b516c0173a27814abe5d8ca60abc8f883ac82b47b5c980877484b"

    if whitelistuser and dvmconfig.IS_BOT:
        user = get_from_sql_table(publickey)
        update_sql_table(user[0], user[1], True, False, user[4], user[5], user[6], user[7])
        user = get_from_sql_table(publickey)
        print(str(user[6]) + " is whitelisted: " + str(user[2]))


    if unwhitelistuser and dvmconfig.IS_BOT:
        user = get_from_sql_table(publickey)
        update_sql_table(user[0], user[1], False, False, user[4], user[5], user[6], user[7])

    if blacklistuser and dvmconfig.IS_BOT:
        user = get_from_sql_table(publickey)
        update_sql_table(user[0], user[1], False, True, user[4], user[5], user[6], user[7])

    if addbalance and dvmconfig.IS_BOT:

        user = get_from_sql_table(publickey)
        update_sql_table(user[0], (int(user[1]) + additional_balance), user[2], user[3], user[4], user[5], user[6], user[7])
        time.sleep(1.0)
        message = str(additional_balance) + " Sats have been added to your balance. Your new balance is " + str((int(user[1]) + additional_balance)) + " Sats."
        keys = Keys.from_sk_str(config.PRIVATE_KEY)
        evt = EventBuilder.new_encrypted_direct_msg(keys, PublicKey.from_hex(publickey) , message,
                                                    None).to_event(keys)
        send_event(evt, key=keys)

    if deleteuser and dvmconfig.IS_BOT:
        delete_from_sql_table(publickey)

    if cleardb and dvmconfig.IS_BOT:
        clear_db()

    if listdatabase:
        list_db()

    if rebroadcast_nip89 and not dvmconfig.IS_BOT:
        nip89_announce_tasks()

def nip89_announce_tasks():

    for nip89 in dvmconfig.NIP89s:
        k_tag = Tag.parse(["k", str(nip89.kind)])
        d_tag = Tag.parse(["d", nip89.dtag])
        keys = Keys.from_sk_str(nip89.pk)
        content = nip89.content
        event = EventBuilder(31990, content, [k_tag, d_tag]).to_event(keys)
        send_event(event, key=keys)

    print("Announced NIP 89")

if __name__ == '__main__':
    os.environ["NOVA_DATA_DIR"] = "W:\\nova\\data"
    os.environ["NOVA_HOST"] = "127.0.0.1"
    os.environ["NOVA_PORT"] = "27017"

    dvmconfig = DVMConfig()
    dvmconfig.PRIVATE_KEY = "privkey"
    dvmconfig.SUPPORTED_TASKS = ["inactive-following", "note-recommendation", "speech-to-text", "summarization",
                                 "translation"]
    dvmconfig.PASSIVE_MODE = True
    nostr_server(dvmconfig)
