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

def load(coordsFolder, imgFolder, firstn = 100, batchSize = 10, 
         shuffle = True, displace = True, show = False, size = 420): 
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
        newsize = imgtensor.shape[1]
        initsize = imgtensor.shape[1]
        if(imgtensor.shape[1] != imgtensor.shape[2]):
            print("image", imgPath, "is not square!")
        
        coordsPath = os.path.join(coordsFolder, coords)
        coordsFile = open(coordsPath, "r")
        # read all coordinates and convert to tensor
        all_coords = []
        
        if displace:
            newsize = size
            x = random.randint(int(0.5 * (imgtensor.shape[2] - newsize)), imgtensor.shape[2] - newsize)
            y = random.randint(0, imgtensor.shape[1] - newsize)
            imgtensor = imgtensor[:, x : x + newsize, y : y + newsize]
            print(newsize)
            newsize = float(newsize) / float(initsize)
            x = float(x) / float(initsize)
            y = float(y) / float(initsize)
            print(newsize, x, y)

        
        for line in coordsFile:
            coords_str = line.strip()
            coords_list = [float(x) for x in coords_str.split()]
            if displace:
                coords_list[0] = (coords_list[0] - y) / newsize
                coords_list[1] = (coords_list[1] - x) / newsize
            all_coords.append(coords_list)
        coords_tensor = torch.tensor(all_coords, dtype=torch.float32)
        coordsFile.close()
        
        if coords_tensor.shape[0] != 68:
            continue
        
        if show:
            import modelutils
            imgfromtensor = modelutils.show_tensor(imgtensor, coords_tensor)
            imgfromtensor.show()
            print(coords_tensor.shape, coords_tensor)

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


def rotate_coords(coords, angle_degrees):
    angle_radians = -torch.deg2rad(torch.tensor(angle_degrees, dtype=torch.float32))
    cos_t = torch.cos(angle_radians)
    sin_t = torch.sin(angle_radians)
    
    rot_mat = torch.tensor([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])

    for i in range(coords.shape[0]):
        x, y = coords[i][0], coords[i][1]
        x -= 0.5
        y -= 0.5
        vecp = torch.tensor([x, y])
        vecp = torch.matmul(rot_mat, vecp)
        coords[i][0] = vecp[0] + 0.5
        coords[i][1] = vecp[1] + 0.5
    return coords

def crop_coords(coords, left, right, top, bottom, prevsize):
    for i in range(coords.shape[0]):
        x, y = coords[i][0] * prevsize, coords[i][1] * prevsize
        x -= left
        y -= top
        coords[i][0] = x / float(right - left)
        coords[i][1] = y / float(bottom - top)
    return coords

def noise_tensor(img_tensor, noise_factor):
    img = img_tensor.numpy().transpose(1, 2, 0)
    noiseframe = np.random.normal(0, noise_factor, img.shape)
    img += noiseframe
    img = np.clip(img, 0, 1)

    noisesize = 3

    downscale = img.copy()
    downscale = cv2.resize(downscale, (0, 0), fx= 1.0 / noisesize, fy= 1.0 / noisesize)
    noiseframe = np.random.normal(0, noise_factor, downscale.shape)
    img += cv2.resize(downscale + noiseframe, (0, 0), fx=noisesize, fy=noisesize)
    img /= 2
    img = np.clip(img, 0, 1)
    
    return torch.from_numpy(img.transpose(2, 0, 1))

def augment_batch(img_tensor_batch, coords_tensor_batch, displace, rotate=0, noise=0):
    imglist, coordlist = [], []
    for img, coords in zip(img_tensor_batch, coords_tensor_batch):
        if(img.shape[1] != img.shape[2]):
            print("image is not square!")
            return None
        
        anglelim = rotate
        angle = np.random.uniform(-anglelim, anglelim)

        mean_color = img.mean(dim=[1, 2])
        img = F.rotate(img, angle, fill=tuple(mean_color))
        coords = rotate_coords(coords, angle)
        
        prevsize = img.shape[1]
        newsize = prevsize - displace
        cornerx = random.randint(0, prevsize - newsize)
        cornery = random.randint(50, (prevsize - newsize))
        prevcoords = coords.clone()
        previmg = img.clone()
        img = img[:, cornery : cornery + newsize, cornerx : cornerx + newsize]
        coords = crop_coords(coords, cornerx, cornerx + newsize, cornery, cornery + newsize, prevsize)
        
        if(torch.max(coords[:, 0]) > 1 or torch.max(coords[:, 1]) > 1 or torch.min(coords[:, 0]) < 0 or torch.min(coords[:, 1]) < 0):
            img = previmg
            coords = prevcoords
            corrective = int((prevsize - newsize) / 3)
            cornerx = random.randint(corrective, prevsize - newsize - corrective)
            cornery = random.randint(corrective, prevsize - newsize - corrective)
            prevcoords = coords.clone()
            previmg = img.clone()
            img = img[:, cornery : cornery + newsize, cornerx : cornerx + newsize]
            coords = crop_coords(coords, cornerx, cornerx + newsize, cornery, cornery + newsize, prevsize)
        
        if(noise > 0):
            img = noise_tensor(img, noise)
        
        imglist.append(img)
        coordlist.append(coords)
    return torch.stack(imglist), torch.stack(coordlist)

def test():   
    train = load("I:/NSU/CV/tests/torch/data/train/coords",
                         "I:/NSU/CV/tests/torch/data/train/images", 
                        firstn = 32, batchSize = 4, shuffle = True,
                        displace = False, show = False, size=512)

    mouth_pointlist = [44, 7, 33, 14, 2, 31, 49, 15, 42, 32, 9, 51, 38, 61,
        18, 23, 12, 47, 67, 1, 2]
    mouth_boundaries = [7, 14, 15, 67]
    eye_L_pointlist = [62, 65, 0, 13, 34, 64]
    eye_R_pointlist = [16, 36, 54, 55, 53, 63]
    nose_pointlist = [5, 29, 51, 30, 4, 57]


    eyes_corners = [53,36,62,13]

    for imgtens, coordtens in train:
        imgtens, coordtens = augment_batch(imgtens, coordtens, displace=80, rotate=30, noise=0.15)
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