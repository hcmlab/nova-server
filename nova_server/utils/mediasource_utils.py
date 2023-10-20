import os
from pytube import YouTube
# [https://overcast.fm/](Overcast) on iOS is my primary podcasting
# listening method, but I have an occasional need to download podcasts for
# archival or offline listening purposes. This script takes advantage of
# Overcast's permalink and showpage to get the podcast author and title
# before downloading the podcast itself from the original page.
#
# Usage: python download_from_overcast.py <overcast_url>
# e.g. python download_from_overcast.py https://overcast.fm/+MWUwqlFc


import re
import sys

from urllib.request import Request, urlopen, urlretrieve
import requests

import nova_server.utils.twitter_video_dl.twitter_video_dl as tvdl


def get_title(html_str):
    """Get the title from the meta tags"""

    title = re.findall(r"<meta name=\"og:title\" content=\"(.+)\"", html_str)
    if len(title) == 1:
        return title[0].replace("&mdash;", "-")
    return None


def get_description(html_str):
    """Get the description from the Meta tag"""

    desc_re = r"<meta name=\"og:description\" content=\"(.+)\""
    description = re.findall(desc_re, html_str)
    if len(description) == 1:
        return description[0]
    return None


def get_url(html_string):
    """Find the URL from the <audio><source>.... tag"""

    url = re.findall(r"<source src=\"(.+?)\"", html_string)
    if len(url) == 1:
        # strip off the last 4 characters to cater for the #t=0 in the URL
        # which urlretrieve flags as invalid
        return url[0][:-4]
    return None


def download_podcast(source_url, target_location):
    """Given a Overcast source URL fetch the file it points to"""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.11 (KHTML, like Gecko) "
        "Chrome/23.0.1271.64 Safari/537.11",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
        "Accept-Encoding": "none",
        "Accept-Language": "en-US,en;q=0.8",
        "Connection": "keep-alive",
    }
    req = Request(source_url, None, headers)
    source_data = urlopen(req).read().decode('utf-8')
    title = get_title(source_data)
    url = get_url(source_data)

    if url is None or title is None:
        sys.exit("Could not find parse URL")
    if not os.path.exists(target_location):
        req = requests.get(url)
        file = open(target_location, 'wb')
        for chunk in req.iter_content(100000):
            file.write(chunk)
        file.close()


def downloadTwitter(videourl, path):
    tvdl.download_video(videourl, path + "twitter.mp4")
    return path + "twitter.mp4"

def downloadTikTok(videourl, path):
    from nova_server.utils.tiktoktinstagram_video_dl.download import TDLALL
    values = [videourl]
    result = TDLALL(values, path)

    return result

def downloadInstagram(videourl, path):
    from nova_server.utils.tiktoktinstagram_video_dl.download import IDL

    result = IDL(videourl, "insta", path)

    return result


def checkYoutubeLinkValid(link):
    try:
        # find a way to test without fully downloading the video
         youtubeObject = YouTube(link)
         youtubeObject = youtubeObject.streams.get_audio_only()
         youtubeObject.download(".", "nostr.youtubetest.mp3")
         os.remove("nostr.youtubetest.mp3")
         return True
    except Exception as e:
        print(str(e))
        return False


def downloadYouTube(link, path, audioonly=True):

    youtubeObject = YouTube(link)
    if audioonly:
        youtubeObject = youtubeObject.streams.get_audio_only()
        youtubeObject.download(path, "nostr.originalaudio.mp3")
        print("Download is completed successfully")
        return path + "nostr.originalaudio.mp3"
    else:
        youtubeObject = youtubeObject.streams.get_highest_resolution()
        youtubeObject.download(path, "nostr.original.mp4")
        print("Download is completed successfully")
        return path + "nostr.original.mp4"






if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("{} <overcast_url>".format(__file__))
    download_podcast(sys.argv[1], "test.mp3")