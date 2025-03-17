import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import random

__all__ = ['augment_batch', 'augment_gen', 'make_filter']

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

    noisesize = 2

    downscale = img.copy()
    downscale = cv2.resize(downscale, (0, 0), fx= 1.0 / noisesize, fy= 1.0 / noisesize)
    noiseframe = np.random.normal(0, noise_factor, downscale.shape)
    img += cv2.resize(downscale + noiseframe, (0, 0), fx=noisesize, fy=noisesize)
    img /= 2
    img = np.clip(img, 0, 1)
    
    return torch.from_numpy(img.transpose(2, 0, 1))

def augment_batch(img_tensor_batch, coords_tensor_batch, device,
                   displace: int, rotate=0, noise=0.0):
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
        # prevcoords = coords.clone()
        # previmg = img.clone()
        img = img[:, cornery : cornery + newsize, cornerx : cornerx + newsize]
        coords = crop_coords(coords, cornerx, cornerx + newsize, cornery,
                            cornery + newsize, prevsize)

        # if(torch.max(coords[:, 0]) > 1 or torch.max(coords[:, 1]) > 1 
        #    or torch.min(coords[:, 0]) < 0 or torch.min(coords[:, 1]) < 0):
        #     img = previmg
        #     coords = prevcoords
        #     corrective = int((prevsize - newsize) / 3)
        #     cornerx = random.randint(corrective, prevsize - newsize - corrective)
        #     cornery = random.randint(corrective, prevsize - newsize - corrective)
        #     prevcoords = coords.clone()
        #     previmg = img.clone()
        #     img = img[:, cornery : cornery + newsize, cornerx : cornerx + newsize]
        #     coords = crop_coords(coords, cornerx, cornerx + newsize,
        #                         cornery, cornery + newsize, prevsize)
        
        if(noise > 0):
            img = noise_tensor(img.to('cpu'), noise).to(device)
        
        imglist.append(img)
        coordlist.append(coords)
    return torch.stack(imglist), torch.stack(coordlist)

def make_filter(*keypoints: list):
    def kpfilter(coordbatch: torch.Tensor):
        return torch.stack([coordbatch[:, k] for k in keypoints]).permute(1, 0, 2)
    return kpfilter

def augment_gen(dataset: list, epochs: int = 1, device = 'cpu', part = 0.8, 
                displace: int = 50, rotate: int = 30, noise: float = 0.1, verbose = False):
    '''
    returns generator of augmented images. 
    dataset - list of pairs of batches (imagebatch, coordsbatch). 
    epochs - number of epochs until end of generation. 
    part - part of the dataset, signed. Example: 0.7 means first 70%, -0.3 means last 30%.  
    displace - how many pixels from side of image we crop randomly.
    rotate - range of random rotations in degrees.
    noise - level of noise, from 0 to 1. Now noise > 0 is slowing generation by 2x 
    because it is estimating on cpu
    '''
    random.shuffle(dataset)
    slicepart = int(len(dataset) * part)
    if verbose:
        print(f'slice of dataset is {slicepart}')
    for i in range(epochs * abs(slicepart)):
        if slicepart > 0:
            bt_images, bt_coords = random.choice(dataset[:slicepart])
        else:
            bt_images, bt_coords = random.choice(dataset[slicepart:])
        bt_images = bt_images.to(torch.float32) / 255
        bt_coords = bt_coords.to(device)
        bt_images = bt_images.to(device)
        if verbose:
            print(f'batch {i} augmented')
            print(f'devices {bt_coords.device}, {bt_images.device}')
        yield augment_batch(bt_images, bt_coords, device, displace, rotate, noise)