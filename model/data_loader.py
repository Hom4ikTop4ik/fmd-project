import os
from PIL import Image
import torch
from torchvision import transforms

import random

def getNumStr(coordsFileName):
    return coordsFileName[-12:-7]

def getImgName(coordsFileName):
    return "dataimg" + getNumStr(coordsFileName) + ".jpeg"

def load(coordsFolder, imgFolder, firstn = 100, batchSize = 10, shuffle = True):
    datalist = []
    batch_img_list = []
    batch_coords_list = []

    coordlist = os.listdir(coordsFolder)
    if(shuffle):
        random.shuffle(coordlist)

    batch_count = 0
    for coords in coordlist:
        if batch_count >= firstn / batchSize:
            break

        imgName = getImgName(coords)
        print(coords, imgName, "batch", 
                batch_count + 1, "from", firstn / batchSize)
        imgPath = os.path.join(imgFolder, imgName)
        img = Image.open(imgPath)
        imgtensor = transforms.ToTensor()(img)
        img.close()

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

        batch_img_list.append(imgtensor)
        batch_coords_list.append(coords_tensor)
        
        if len(batch_img_list) == batchSize:
            batch_img_tensor = torch.stack(batch_img_list)
            batch_coords_tensor = torch.stack(batch_coords_list)
            datalist.append((batch_img_tensor, batch_coords_tensor))
            batch_img_list = []
            batch_coords_list = []
            batch_count += 1

    return datalist 