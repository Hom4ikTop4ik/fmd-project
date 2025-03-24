import cv2
import os
import sys
import signal
import torch
import torch.nn as nn
import numpy as np

from data_process import augment_gen
from data_process import MakerPCA
from data_process import load

def gen_lim(generator, limit):
    for i, obj in enumerate(generator):
        if i >= limit:
            break
        yield obj

# establishing paths and loading
current_path = os.path.dirname(os.path.abspath(__file__))
registry_path = os.path.join(current_path, 'registry')
weight_save_path = os.path.join(registry_path, 'weights', 'model_bns.pth')
dataset = load(500, 40, os.path.join(current_path, registry_path, 'dataset'), imagesfile='dataset_coords_ext.pt')

# establishing devices and signal handler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("devise is: ", device)

mypca = MakerPCA().fit(gen_lim(augment_gen(dataset, epochs=10, device=device,
                                        noise=0, part=0.9, displace=80, rotate=20, verbose=True), 200), 40, verbose=True)

mypca.save(os.path.join(current_path,'data_process/pcaweights_ext.pca'))