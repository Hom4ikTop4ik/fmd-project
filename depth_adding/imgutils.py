from PIL import Image, ImageDraw
import cv2
import coordsParser

def showPointsWithDepth(imgfile:str, coordsfile:str, depthList):
    img = cv2.imread(imgfile)
    rectsize = 4
    coordList = coordsParser.parse(coordsfile)
    
    for midv, depth in zip(coordList, depthList):
        rectangl = [(midv[0] - rectsize, midv[1] - rectsize), 
                    (midv[0] + rectsize, midv[1] + rectsize) ]
        cv2.circle(img, (midv[0], midv[1]), 5, (0, 255, 0), -1)
        cv2.putText(img, str(depth)[:4], (midv[0], midv[1]),
                    1, 1.2, (255,255,255), 2, 2)
        # exVisDraw.rectangle(rectangl, (255,255,255))
        # img.putText()
        # exVisDraw.text(midv, str(depth)[:6],)
    img = cv2.resize(img, (720,720))
    cv2.imshow("face depth", img)
    cv2.waitKeyEx(0)


def normalizeCoords(imageSize, coordPair):
    for i in range(len(imageSize)):  
        coordPair[i] /= imageSize[i]
    print(coordPair)
    return coordPair