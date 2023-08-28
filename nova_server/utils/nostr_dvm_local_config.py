import os
from threading import Thread

from nova_server.utils.nostr_dvm import DVMConfig, nostr_server, NIP89Announcement

def run_nostr_dvm_with_local_config():

    MACHINE_SUPPORTED_TASKS = ["inactive-following", "note-recommendation", "speech-to-text", "image-to-text",
                                 "summarization", "translation"]


    dvmconfig = DVMConfig()
    dvmconfig.PRIVATE_KEY = os.environ["NOVA_NOSTR_KEY"]
    dvmconfig.SUPPORTED_TASKS = MACHINE_SUPPORTED_TASKS
    dvmconfig.IS_BOT = False

    #Add NIP89 Announcements, the DVM might submit
    nio89textextraction = NIP89Announcement()
    nio89textextraction.kind = 65002
    nio89textextraction.dtag = "0wllfdaxzp624bji"
    nio89textextraction.pk = os.environ["NOVA_NOSTR_KEY"]
    nio89textextraction.content = "{\"name\":\"NostrAI DVM Text Extractor\",\"image\":\"https://cdn.nostr.build/i/38f36583552828a0961b01cddc6a3f4cd3ed8250dc5f73ab22ed7ed03eceaed9.jpg\",\"about\":\"Providing results using WhisperX model for the input formats: wav, mp3, mp4, ogg, avi, mov, as well as youtube, and overcast links. \\nPossible outputs: text/plain,  empty output format will provide timestamped labels with granularity depending on alignment parameters (word, segment, raw).\\nDefault model: base, Default alignment: raw\\n\\nFurther allows text extraction from images with tesseract OCR for jpgs.\",\"nip90Params\":{\"model\":{\"required\":false,\"values\":[\"tiny\",\"base\",\"small\",\"tiny.en\",\"base.en\",\"small.en\"]},\"alignment\":{\"required\":false,\"values\":[\"word\",\"segment\",\"raw\"]}}}"
    dvmconfig.NIP89s.append(nio89textextraction)

    nip89summarization = NIP89Announcement()
    nip89summarization.kind = 65003
    nip89summarization.dtag = "ns5x24xqm03vuiw4"
    nip89summarization.pk = os.environ["NOVA_NOSTR_KEY"]
    nip89summarization.content = "{\"name\":\"NostrAI Summarizer\",\"image\":\"https://cdn.nostr.build/i/a177be1159da5aad8396a1188f686728d55647d3a7371549584daf2b5e50eec9.jpg\",\"about\":\"Uses a LLAMA2 instance to summarise the most important points of a given input (text, event, job). Can for example be applied on transcribed podcasts, Nostr Long form events, etc.\",\"nip90Params\":{}}"
    dvmconfig.NIP89s.append(nip89summarization)

    nip89translation = NIP89Announcement()
    nip89translation.kind = 65004
    nip89translation.dtag = "dpsu1wsh7ubsioy2"
    nip89translation.pk = os.environ["NOVA_NOSTR_KEY"]
    nip89translation.content = "{\"name\":\"NostrAI DVM Translator\",\"image\":\"https://cdn.nostr.build/i/feb98d8700abe7d6c67d9106a72a20354bf50805af79869638f5a32d24a5ac2a.jpg\",\"about\":\"Translates Text from given text/event/job, currently using Google Translation Services into language defined in param.  \",\"nip90Params\":{\"language\":{\"required\":true,\"values\":[\"af\",\"am\",\"ar\",\"az\",\"be\",\"bg\",\"bn\",\"bs\",\"ca\",\"ceb\",\"co\",\"cs\",\"cy\",\"da\",\"de\",\"el\",\"eo\",\"es\",\"et\",\"eu\",\"fa\",\"fi\",\"fr\",\"fy\",\"ga\",\"gd\",\"gl\",\"gu\",\"ha\",\"haw\",\"hi\",\"hmn\",\"hr\",\"ht\",\"hu\",\"hy\",\"id\",\"ig\",\"is\",\"it\",\"he\",\"ja\",\"jv\",\"ka\",\"kk\",\"km\",\"kn\",\"ko\",\"ku\",\"ky\",\"la\",\"lb\",\"lo\",\"lt\",\"lv\",\"mg\",\"mi\",\"mk\",\"ml\",\"mn\",\"mr\",\"ms\",\"mt\",\"my\",\"ne\",\"nl\",\"no\",\"ny\",\"or\",\"pa\",\"pl\",\"ps\",\"pt\",\"ro\",\"ru\",\"sd\",\"si\",\"sk\",\"sl\",\"sm\",\"sn\",\"so\",\"sq\",\"sr\",\"st\",\"su\",\"sv\",\"sw\",\"ta\",\"te\",\"tg\",\"th\",\"tl\",\"tr\",\"ug\",\"uk\",\"ur\",\"uz\",\"vi\",\"xh\",\"yi\",\"yo\",\"zh\",\"zu\"]}}}"
    dvmconfig.NIP89s.append(nip89translation)

    # nip89imagegeneration = NIP89Announcement()
    # nip89imagegeneration.kind = 65005
    # nip89imagegeneration.dtag = "06sfjfp9frr3ubcq"
    # nip89imagegeneration.pk = os.environ["NOVA_NOSTR_KEY"]
    # nip89imagegeneration.content = "{\"name\":\"NostrAI DVM Artist\",\"image\":\"https://cdn.nostr.build/i/2c9ff28899732291fdcde742747b533a12c56185a345ce94c0b9e5ae9f5460f8.jpg\",\"about\":\"Generate an Image based on a prompt. Supports various models. By default uses Stable Diffusion XL 1.0. \\nPossible Inputs are text, events or jobs. Lora (Specific weights) param only works for SDXL.\\nAn optional negative prompt can help the model avoid things it shouldn't do.\\nImages are upscaled  4x by default.\\n\\nAdditionally supports Image2Image conversion. Requires as input url of an image/previous job/event and a second text input containing the prompt. By default, uses instruct-pix2pix model, alternative is sdxl (Stable Diffusion XL) model.\",\"nip90Params\":{\"model\":{\"required\":false,\"values\":[\"sdxl\",\"dreamshaper\",\"nightvision\",\"protovision\",\"dynavision\",\"sdvn\",\"wild\",\"realistic\",\"lora_inks\",\"lora_pepe\"]},\"ratio\":{\"required\":false,\"values\":[\"1:1\",\"4:3\",\"16:9\",\"16:10\",\"3:4\",\"9:16\",\"10:16\"]},\"negative_prompt\":{\"required\":false,\"values\":[]},\"extra_prompt\":{\"required\":false,\"values\":[]},\"upscale\":{\"required\":false,\"values\":[\"1\",\"2\",\"3\",\"4\"]},\"lora\":{\"required\":false,\"values\":[\"3d_render_style_xl\",\"cyborg_style_xl\",\"psychedelic_noir_xl\",\"dreamarts_xl\",\"voxel_xl\",\"kru3ger_xl\",\"wojak_xl\"]}}}"
    # dvmconfig.NIP89s.append(nip89imagegeneration)

    nip89filterevents = NIP89Announcement()
    nip89filterevents.kind = 65007
    nip89filterevents.dtag = "ebfw1pwoe2cx2f7n"
    nip89filterevents.pk = os.environ["NOVA_NOSTR_KEY"]
    nip89filterevents.content = "{\"name\":\"Nostr AI DVM Inactive Followings\",\"image\":\"https://cdn.nostr.build/i/fff3f825ff3aa20daf0cb6e099264dfd5b7a66b0922431d22810b33e8de13d36.jpg\",\"about\":\"Returns a list of inactive followings. This includes npubs of users who haven't posted or reacted within the last x days (default 30). Parameter since can be used to increase the search window, e.g. inactive in the last 60 days.\",\"nip90Params\":{\"since\":{\"required\":false,\"values\":[\"30\",\"60\",\"90\",\"120\",\"150\",\"180\"]}}}"
    dvmconfig.NIP89s.append(nip89filterevents)

    nostr_dvm_thread = Thread(target=nostr_server, args=[dvmconfig])
    nostr_dvm_thread.start()


    # BOT MIGHT RUN ITS OWN PK
    dvmconfig = DVMConfig()
    dvmconfig.PRIVATE_KEY = os.environ["NOVA_NOSTR_KEY"]
    dvmconfig.SUPPORTED_TASKS = MACHINE_SUPPORTED_TASKS
    dvmconfig.IS_BOT = True
    dvmconfig.PASSIVE_MODE = True

    nostr_bot_thread = Thread(target=nostr_server, args=[dvmconfig])
    nostr_bot_thread.start()