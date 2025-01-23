from augments import augmented_gen
from data_loader import load
import os
import datetime
import numpy as np
import cv2

def test():   
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print('load started', datetime.datetime.now().time())
    train = load(1000, 20, os.path.join(current_dir, '../registry/dataset'))
    print('load ended', datetime.datetime.now().time())
    gen = augmented_gen(1, train)
    for imgtens, coordtens in gen:
        
        coords = coordtens[0]

        newimg = (imgtens[0,[2,1,0], :, :].numpy().transpose(1,2,0) * 255).astype(np.uint8)

        newimg = np.ascontiguousarray(newimg)
        print(newimg.shape, np.max(newimg), np.min(newimg), newimg)
        for coord in coords:
            x, y = coord[0], coord[1]
            newimg = cv2.circle(newimg, (int(x * newimg.shape[1]), int(y * newimg.shape[0])), 2, (255, 255, 255), 2)
        
        cv2.imshow("img", newimg)
        cv2.waitKey(0)

test()