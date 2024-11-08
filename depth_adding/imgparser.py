import numpy as np
from PIL import Image

def parse(img_marks:Image.Image):
    arr = img_marks.histogram()
    sizex, sizey = img_marks.size
    colorids = []
    for i in range(0, 256):
        if arr[1 * i] > 0 or arr[2 * i] > 0 or arr[3 * i]:
            colorids.append((1 * i,  2 * i, arr[3 * i]))    
    s = set()
    for x in range(0, img_marks.width, 4):
        for y in range(0, img_marks.height, 4):
            s.add(img_marks.getpixel((x, y)))
    facevecs = []
    for el in s:
        if el == (0,0,0):
            continue
        midvec = (0, 0)
        cnteqs = 0
        for x in range(0, img_marks.width, 4):
            for y in range(0, img_marks.height, 4):
                if el == img_marks.getpixel((x, y)) :
                    cnteqs += 1
                    midvec = midvec[0] + x, midvec[1] + y
        facevecs.append(np.array([midvec[0] / cnteqs / sizex, midvec[1] / cnteqs / sizey]))
    return facevecs
