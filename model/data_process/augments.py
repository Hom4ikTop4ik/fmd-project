import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import random
import threading
import time

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

def show_image(img_to_print, test = 0):
    name = "img" + str(random.randint(0, 1000))
    cv2.imshow(name, img_to_print)
    cv2.waitKey(test)
    # cv2.destroyWindow(name) # Закрываем окно после отображения

def noise_tensor(img_tensor, noise_factor, verbose = False):
    device = img_tensor.device # Сохраняем текущее устройство
    if verbose:
        print("augments.py/noise_tensor/device:" + str(device))

    # Добавляем гауссовский шум
    noise = torch.randn_like(img_tensor, device=device) * noise_factor
    img_tensor = torch.clamp(img_tensor + noise, 0, 1)
    
    # Во сколько раз уменьшать картинку, список значений.
    # Дальше берётся среднее арифметическое.
    noise_sizes = [2, 4, 8] 

    for noise_size in noise_sizes:
        # Уменьшаем изображение
        downscale = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0), # создать псевдо ОСь с размером батча 
            scale_factor=1.0 / noise_size, mode='bilinear', align_corners=False
        ).squeeze(0) # отбросить псевдо ось

        # Добавляем шум в уменьшенное изображение
        noise_small = torch.randn_like(downscale, device=device) * noise_factor
        downscale = torch.clamp(downscale + noise_small, 0, 1)
    
        # Увеличиваем обратно
        upscaled = torch.nn.functional.interpolate(
            downscale.unsqueeze(0), scale_factor=noise_size, mode='bilinear', align_corners=False
        ).squeeze(0)

        img_tensor = img_tensor + upscaled

    img_tensor = torch.clamp(img_tensor / (len(noise_sizes) + 1), 0, 1)
    

    if (random.random() < 0.005):
        test = 1
        show_image(img_tensor.cpu().numpy().transpose(1, 2, 0), test)
    #     thread = threading.Thread(target=show_image, args=(img_tensor.cpu().numpy().transpose(1, 2, 0),))
    #     # print(thread, end="")
    #     thread.start()

    return img_tensor

def augment_batch(img_tensor_batch, coords_tensor_batch, device,
                   displace: int, rotate=0, noise=0.0):
    imglist, coordlist = [], []
    for img, coords in zip(img_tensor_batch, coords_tensor_batch):
        if(img.shape[1] != img.shape[2]):
            print("image is not square!")
            return None
        
        img = img[[2,1,0],:,:] # Каналы обратно вернуть

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
            img = noise_tensor(img, noise)
        
        imglist.append(img)
        coordlist.append(coords)
    return torch.stack(imglist), torch.stack(coordlist)

def make_filter(*keypoints: list):
    def kpfilter(coordbatch: torch.Tensor):
        return torch.stack([coordbatch[:, k] for k in keypoints]).permute(1, 0, 2)
    return kpfilter

def augment_gen(dataset: list, epochs: int = 1, device = 'cuda:0' if torch.cuda.is_available() else 'cpu', part = 0.8, 
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
