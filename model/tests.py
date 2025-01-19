import cv2
import numpy as np

def coords_to_img(coords, idlist, imgsize):
    midx = 0.0
    midy = 0.0
    for id in idlist:
        midx += coords[id][0]
        midy += coords[id][1]
    midx /= len(idlist)
    midy /= len(idlist)
    
    img = np.zeros((imgsize, imgsize), dtype=np.float32)
    x = int(midx)
    y = int(midy)
    print(x, y)
    img[y, x] = 15000
    img = cv2.GaussianBlur(img, (351, 351), sigmaX=50, borderType=cv2.BORDER_REPLICATE)
    cv2.circle(img, (x, y), 0, 6, -1)
    img = cv2.GaussianBlur(img, (11, 11), sigmaX=1)
    return img

def test_coords_to_img():
    coords = [(0,0), (200,130)]
    img = coords_to_img(coords, [0,1], 400)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("img", img)
    cv2.waitKey(0)

test_coords_to_img()
