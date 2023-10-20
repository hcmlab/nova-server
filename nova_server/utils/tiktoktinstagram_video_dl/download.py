import requests

import instaloader

def getDict() -> dict:
    response = requests.get('https://ttdownloader.com/')
    point = response.text.find('<input type="hidden" id="token" name="token" value="') + \
        len('<input type="hidden" id="token" name="token" value="')
    token = response.text[point:point+64]
    TTDict = {
        'token': token,
    }

    for i in response.cookies:
        TTDict[str(i).split()[1].split('=')[0].strip()] = str(
            i).split()[1].split('=')[1].strip()
    return TTDict



def createHeader(parseDict) -> list:

    cookies = {
        'PHPSESSID': parseDict['PHPSESSID'],
        # 'popCookie': parseDict['popCookie'],
    }
    headers = {
        'authority': 'ttdownloader.com',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'origin': 'https://ttdownloader.com',
        'referer': 'https://ttdownloader.com/',
        'sec-ch-ua': '"Not?A_Brand";v="8", "Chromium";v="108", "Google Chrome";v="108"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest',
    }
    data = {
        'url': '',
        'format': '',
        'token': parseDict['token'],
    }
    return cookies, headers, data


def TDL(cookies, headers, data, name, path) -> str:
    response = requests.post('https://ttdownloader.com/search/',
                             cookies=cookies, headers=headers, data=data)
    linkParse = [i for i in str(response.text).split()
                 if i.startswith("href=")][0]

    response = requests.get(linkParse[6:-10])
    with open(path +"tiktok"+name+".mp4", "wb") as f:
        f.write(response.content)
    return path +"tiktok"+name+".mp4"


def TDLALL(linkList, path) -> str:
    parseDict = getDict()
    cookies, headers, data = createHeader(parseDict)
    #linkList = getLinkDict()['tiktok']
    for i in linkList:
        try:
            data['url'] = i
            result = TDL(cookies, headers, data, str(linkList.index(i)), path)
            return result
        except IndexError:
            parseDict = getDict()
            cookies, headers, data = createHeader(parseDict)
        except Exception as err:
            print(err)
            exit(1)


def IDL(url, name, path) -> str:

    obj = instaloader.Instaloader()
    post = instaloader.Post.from_shortcode(obj.context, url.split("/")[-2])
    photo_url = post.url
    video_url = post.video_url
    print(video_url)
    if video_url:
        response = requests.get(video_url)
        with open(path +"insta"+name+".mp4", "wb") as f:
            f.write(response.content)
            return path +"insta"+name+".mp4"
    elif photo_url:
        response = requests.get(photo_url)
        with open(path +"insta" +name+".jpg", "wb") as f:
            f.write(response.content)
            return path +"insta" +name+".jpg"



def IDLALL(linklist, path) -> str:
    for i in linklist:
        try:
            print(str(linklist.index(i)))
            print(str(linklist[i]))
            result = IDL(i, str(linklist.index(i)), path)
            return result
        except Exception as err:
            print(err)
            exit(1)


if __name__ == "__main__":
    # TDLALL()
    # IDLALL()
    pass
