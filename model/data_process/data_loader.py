import torch
import os
import cv2
import numpy as np
import re

from torch.utils.data import Dataset, DataLoader
from data_process.__init__ import noise, rotate, min_scale, max_scale, blur_level
from data_process.augments import augment_image, scale_img, show_image
from data_process.__init__ import USE_CPU_WHATEVER, DA, NET, BATCH_SIZE, POCHTI

from torch.utils.data import Sampler
import random

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

        if self.aug  or  self.aug == POCHTI:
            scale = torch.FloatTensor(1).uniform_(min_scale, max_scale).item()
            image, coords = augment_image(image, coords, rotate, noise, scale, blur_level, self.aug)

        # image = image.to(self.device)
        # coords = coords.to(self.device)

        return image, coords

def collate_fn(batch):
    images, coords = zip(*batch)
    images = torch.stack(images, dim=0)  # Собираем батч
    coords = torch.stack(coords, dim=0)
    return images, coords

def load(
        bsize: int = BATCH_SIZE, 
        dataset_path: str = '.', 
        images_dir: str = 'images', 
        coords_dir: str = 'coords',
        device = 'cpu',
        sampler_seed = 0,
        augments = DA,
        workers = DA
    ):
    images_dir_full = os.path.join(dataset_path, images_dir)
    coords_dir_full = os.path.join(dataset_path, coords_dir)
    
    # augments = DA
    dataset = CustomDataset(images_dir_full, coords_dir_full, device, augments)
    sampler = EpochShuffleSampler(len(dataset), sampler_seed)
    
    # 8 workers and prefetch=3 is about 3GB video memory

    num_workers = 8
    if not workers:
        num_workers = 0

    pin = True # in start device is cpu, so memory_pin is avaiable

    # Используем prefetch_factor для предзагрузки данных
    prefetch_factor = 2
    if (prefetch_factor <= 0) or (num_workers <= 0):
        prefetch_factor = None

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
