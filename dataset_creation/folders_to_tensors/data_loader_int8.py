from hmac import new
import os
from PIL import Image
import cv2
from numpy import eye
import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random

def getNumStr(coordsFileName):
    return coordsFileName[-12:-7]

def getImgName(coordsFileName):
    return "dataimg" + getNumStr(coordsFileName) + ".jpeg"

def load(coordsFolder, imgFolder, fromid, toid): 
    coords_t_list = []
    images_t_list = []
    coordlist = os.listdir(coordsFolder)
    # if(shuffle):
        # random.shuffle(coordlist)
    coordlist = coordlist[fromid:toid]
    print(len(coordlist))

    for i, coords in enumerate(coordlist):
        print(i)
        imgName = getImgName(coords)
        imgPath = os.path.join(imgFolder, imgName)
        img = Image.open(imgPath)
        imgtensor = transforms.ToTensor()(img)
        img.close()
        if(imgtensor.shape[1] != imgtensor.shape[2]):
            print("image", imgPath, "is not square!")
        
        coordsPath = os.path.join(coordsFolder, coords)
        coordsFile = open(coordsPath, "r")
        # read all coordinates and convert to tensor
        all_coords = []
        
    
        
        for line in coordsFile:
            coords_str = line.strip()
            coords_list = [float(x) for x in coords_str.split()]
            all_coords.append(coords_list)
        coords_tensor = torch.tensor(all_coords, dtype=torch.float32)
        coordsFile.close()
        
        if coords_tensor.shape[0] != 68:
            continue
        
        coords_t_list.append(coords_tensor)
        images_t_list.append((imgtensor * 255).to(torch.uint8))

    return torch.stack(images_t_list), torch.stack(coords_t_list) 