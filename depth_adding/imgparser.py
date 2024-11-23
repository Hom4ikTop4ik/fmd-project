import numpy as np
from PIL import Image
import mappings
import cv2
import numpy as np

def showPointsWithIndexes(imgfile:str, coordList):
    img = cv2.imread(imgfile)
    i = 0
    for midv in coordList:
        if(1):
            cv2.circle(img, (midv[0], midv[1]), 2, (0, 255, 0), -1)
            cv2.putText(img, str(i)[:4], (midv[0], midv[1]),
                    1, 0.5, (255,255,255), 1, 2)
        i+=1

    cv2.imshow("face depth", img)
    cv2.waitKeyEx(0)


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
    facevecsmap = {}
    facevecs = []
    i = 0
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
        facevecsmap[el] = i
        facevecs.append(np.array([midvec[0] / cnteqs / sizex, midvec[1] / cnteqs / sizey]))
        i += 1
    #return facevecs, facevecsmap
    if len(facevecs) == 68:
        return facevecs, facevecsmap
    else:
        
        # check for the corners of lips, if there are missing ones, restore it from another corner
        # get corners ids from mappings.py, check for the color, if color is not equal to
        # that in mappings.py, corner of lip is missing

        lipRInClr = mappings.IdToColorMapping[mappings.lipRightInner]
        lipROutClr = mappings.IdToColorMapping[mappings.lipRightOut]
        lipLOutClr = mappings.IdToColorMapping[mappings.lipLeftOut]
        lipLInClr = mappings.IdToColorMapping[mappings.lipLeftInner]
        if lipRInClr not in facevecsmap.keys():
            facevecsmap[lipRInClr] = mappings.lipRightInner
            facevecs.insert(mappings.lipRightInner, facevecs[facevecsmap[lipROutClr]])
        if lipLOutClr not in facevecsmap.keys():
            facevecsmap[lipLOutClr] = mappings.lipLeftOut
            facevecs.insert(mappings.lipLeftOut, facevecs[facevecsmap[lipLInClr]])
        if lipLInClr not in facevecsmap.keys():
            facevecsmap[lipLInClr] = mappings.lipLeftInner
            facevecs.insert(mappings.lipLeftInner, facevecs[facevecsmap[lipLOutClr]])
        if lipROutClr not in facevecsmap.keys():
            facevecsmap[lipROutClr] = mappings.lipRightOut
            facevecs.insert(mappings.lipRightOut, facevecs[facevecsmap[lipRInClr]])

        return facevecs, facevecsmap



# if no some color exist, point is missing, we'll change it to the closest 

filename = "./samples/circles/condimg3.jpeg"
fileimage = "./samples/circimages/img3.jpeg"
img = Image.open(filename)
veclist, vecmap = parse(img)
newlist = []
for el in veclist:
    pair = [int(el[0] * 512)]
    pair.append(int(el[1] * 512))
    newlist.append(pair)

print(len(newlist))

showPointsWithIndexes(fileimage, newlist)
