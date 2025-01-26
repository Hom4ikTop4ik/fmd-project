from augments import augment_gen, make_filter
from data_loader import load
import os
import torch
import datetime
import numpy as np
import cv2

def test():   
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print('load started', datetime.datetime.now().time())
    train = load(1000, 20, os.path.join(current_dir, '../registry/dataset'))
    print('load ended', datetime.datetime.now().time())
    time = datetime.datetime.now()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen = augment_gen(train, epochs=1, device=device, noise=0.1)

    extractor = make_filter(53, 36, 62, 13, 14, 30, 44)
    for imgtens, coordtens in gen:
        # coords = extractor(coordtens)[0]

        # newimg = (imgtens[0,[2,1,0], :, :].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

        # newimg = np.ascontiguousarray(newimg)
        # for coord in coords:
        #     x, y = coord[0], coord[1]
        #     newimg = cv2.circle(newimg, (int(x * newimg.shape[1]), int(y * newimg.shape[0])), 2, (255, 255, 255), 2)
        
        # cv2.imshow("img", newimg)
        # cv2.waitKey(0)
        print('augmented', datetime.datetime.now() - time)
        time = datetime.datetime.now()

test()