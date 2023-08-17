import os


class DVMConfig:
    #SUPPORTED_TASKS = ["inactive-following", "speech-to-text", "summarization", "translation"]
    SUPPORTED_TASKS = ["text-to-image", "image-to-image", "image-upscale","image-to-text"]
    PASSIVE_MODE: bool = False  # instance should only do tasks set in SUPPORTED_TASKS, no bot chatting, manage zaps etc
    USERDB = "W:\\nova\\tools\\AnnoDBbackup\\nostrzaps.db"
    RELAY_LIST = ["wss://relay.damus.io", "wss://blastr.f7z.xyz", "wss://nostr-pub.wellorder.net", "wss://nos.lol",
                  "wss://nostr.wine", "wss://relay.nostr.com.au", "wss://relay.snort.social"]
    RELAY_TIMEOUT = 1
    LNBITS_INVOICE_KEY = 'bfdfb5ecfc0743daa08749ce58abea74'
    LNBITS_URL = 'https://lnbits.novaannotation.com'
    REQUIRES_NIP05: bool = False

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


DVMConfig()