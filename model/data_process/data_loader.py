import torch
import os
import cv2
import numpy as np
import re

from torch.utils.data import Dataset, DataLoader
from data_process.__init__ import noise, rotate, min_scale, max_scale
from data_process.augments import augment_image
from data_process.__init__ import USE_CPU_WHATEVER, DA, NET

from torch.utils.data import Sampler
import random

# BDD doesn't know what is it, but look like very important
mapping_h2f = {
    37: 0,
    66: 1,
    27: 2,
    41: 3,
    58: 4,
    17: 5,
    39: 6,
    48: 7,
    6: 8,
    21: 9,
    8: 10,
    52: 11,
    22: 12,
    25: 13,
    43: 14,
    40: 15,
    24: 16,
    20: [25, 26],
    59: 24,
    3: 23,
    26: 22,
    56: 21,
    10: 20,
    28: 19,
    60: [17, 18],
    53: 36,
    55: 37,
    54: 38,
    36: 39,
    16: 40,
    63: 41,
    62: 42,
    64: 43,
    34: 44,
    13: 45,
    0: 46,
    65: 47,
    5: 27,
    29: 28,
    51: 29,
    30: 30,
    4: 31,
    46: 32,
    45: 33,
    35: 34,
    57: 35,
    44: 48,
    7: 60,
    31: 49,
    49: 50,
    15: 51,
    42: 52,
    32: 53,
    14: 64,
    33: 54,
    2: 55,
    1: 56,
    67: 57,
    47: 58,
    12: 59,
    9: 61,
    50: 62,
    38: 63,
    23: 65,
    18: 66,
    61: 67
}

def depth_permute(coords: torch.Tensor):
    oldcord = coords.clone()
    for key in mapping_h2f.keys():
        if type(mapping_h2f[key]) == list:
            for el in mapping_h2f[key]:
                coords[key][2] = oldcord[el][2]
        else:
            coords[key][2] = oldcord[mapping_h2f[key]][2]
    return coords



class EpochShuffleSampler(Sampler):
    def __init__(self, data_len, seed=0):
        self.data_len = data_len
        self.seed = seed

    def set_seed(self, seed):
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        indices = torch.randperm(self.data_len, generator=g).tolist()
        return iter(indices)

    def __len__(self):
        return self.data_len

class CustomDataset(Dataset):
    def __init__(self, images_dir, coords_dir, device, aug):
        self.images_dir = images_dir
        self.coords_dir = coords_dir
        self.device = device
        self.aug = aug

        self.coords_files = [f for f in os.listdir(coords_dir) if f.endswith('_3d.txt')]
        if not self.coords_files:
            raise FileNotFoundError(f"No coordinate files found in {coords_dir}.")

        self.image_files = []

        for coords_file in self.coords_files:
            match = re.search(r'\d+', coords_file)
            numeric_part = match.group(0)

            image_file = f"dataimg{numeric_part}.jpeg"
            img_path = os.path.join(images_dir, image_file)            
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image {img_path} does not exist.")
            self.image_files.append(image_file)
            
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = torch.from_numpy(cv2.imread(img_path, cv2.IMREAD_UNCHANGED)).float() / 255.0  
        image = image.permute(2, 0, 1) # to (channels, height, width)
        
        coords_path = os.path.join(self.coords_dir, self.coords_files[idx])
        coords = torch.from_numpy(np.loadtxt(coords_path)).float()

        if self.aug:
            scale = torch.FloatTensor(1).uniform_(min_scale, max_scale).item()
            image, coords = augment_image(image, coords, rotate, noise, scale)

        # image = image.to(self.device)
        # coords = coords.to(self.device)

        # if (DA):
            # permuted_coords = depth_permute(coords)
            # print("\t\tcoords permuted")
            # return image, permuted_coords

        return image, coords

def collate_fn(batch):
    images, coords = zip(*batch)
    images = torch.stack(images, dim=0)  # Собираем батч
    coords = torch.stack(coords, dim=0)
    return images, coords

def load(
        bsize: int = 40, 
        dataset_path: str = '.', 
        images_dir: str = 'images', 
        coords_dir: str = 'coords',
        device = 'cpu',
        sampler_seed = 0
    ):
    print(device)
    images_dir_full = os.path.join(dataset_path, images_dir)
    coords_dir_full = os.path.join(dataset_path, coords_dir)
    
    augments = DA
    dataset = CustomDataset(images_dir_full, coords_dir_full, device, augments)
    sampler = EpochShuffleSampler(len(dataset), sampler_seed)
    
    # 8 workers and prefetch=3 is about 3GB video memory

    num_workers = 8
    prefetch_factor = 0 if (num_workers > 0) else None 
    if prefetch_factor == 0:
        prefetch_factor = None
    pin = True # not torch.cuda.is_available() or USE_CPU_WHATEVER

    # Используем prefetch_factor для предзагрузки данных
    dataloader = DataLoader(
        dataset,
        batch_size=bsize,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        # shuffle=True,
        sampler=sampler
    )

    print(f"\n\tnum_workers: {num_workers}, pin_memory: {pin}, persistent_workers: {num_workers > 0}, prefetch_factor: {prefetch_factor}\n")
    return dataloader, sampler
