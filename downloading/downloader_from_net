import requests
from bs4 import BeautifulSoup
import os

dirname = "outssimages"
prefix = "a"
imageCount = 10000
if dirname not in os.listdir():
    os.mkdir(dirname)

for i in range(0, imageCount):
    try:
        page = requests.get("https://thispersondoesnotexist.com/")
        souped = BeautifulSoup(page.content, "html.parser")
        file = open("./" + dirname + "/" + prefix + "_img" + str(i) + ".jpeg", 'wb')
        file.write(page.content)
        file.close()
    except:
        print("i almost falled, but arised again and continue image getting\n")
        continue
