import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import re  

class CustomDataset(Dataset):
    def __init__(self, images_dir, coords_dir):
        """
        images_dir: путь к папке с изображениями.
        coords_dir: путь к папке с координатами.
        """
        self.images_dir = images_dir
        self.coords_dir = coords_dir

        self.coords_files = sorted([f for f in os.listdir(coords_dir) if f.endswith('_3d.txt')])
        if not self.coords_files:
            raise FileNotFoundError(f"No coordinate files found in {coords_dir}.")

        self.image_files = []
        for coords_file in self.coords_files:
            match = re.search(r'\d+', coords_file)
            numeric_part = match.group(0)

            image_file = f"dataimg{numeric_part}.jpeg"
            self.image_files.append(image_file)

        for img_file in self.image_files:
            img_path = os.path.join(images_dir, img_file)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image {img_path} does not exist.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = torch.from_numpy(cv2.imread(img_path)).float() / 255.0  
        image = image.permute(2, 0, 1) 

        coords_path = os.path.join(self.coords_dir, self.coords_files[idx])
        coords = torch.from_numpy(np.loadtxt(coords_path)).float()

        return image, coords

def load(
        bsize: int = 40, 
        #change on real path 
        dataset_path: str = '.', 
        images_dir: str = 'images', 
        coords_dir: str = 'coords'
    ):

    images_dir_full = os.path.join(dataset_path, images_dir)
    coords_dir_full = os.path.join(dataset_path, coords_dir)
    
    dataset = CustomDataset(images_dir_full, coords_dir_full)
    dataloader = DataLoader(dataset, batch_size=bsize, shuffle=True, num_workers=4)

    return dataloader
