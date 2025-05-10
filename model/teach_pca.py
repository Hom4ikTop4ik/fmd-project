import cv2
import os
import sys
import signal
import time
import torch
import torch.nn as nn
import numpy as np

# from data_process import augment_gen
# from data_process import MakerPCA
from data_process import load

from data_process.convertor_pca import MakerPCA, PCA_COUNT
from detector.blocked import MultyLayer


def gen_lim(limit, generator):
    for i, obj in enumerate(generator):
        # if (i % 1 == 0) :
        print(i)
        if i >= limit:
            break
        yield obj

# establishing paths and loading
current_path = os.path.dirname(os.path.abspath(__file__))
registry_path = os.path.join(current_path, 'registry')
# dataset = load(500, 40, os.path.join(current_path, registry_path, 'dataset'), imagesfile='dataset_coords_ext.pt')

# establishing devices and signal handler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(time.time())
print("devise is: ", device)
if __name__ == "__main__":
    dataLoader, sampler = load(40, "./registry/dataset/train", "images", "extended_coords", device, sampler_seed=18)

    dataIterator = iter(dataLoader)
    print("AHAHHA")


    again = input("Train yet? (y/n)")
    mypca = MakerPCA()
    if (again != "n"):
        print("load")
        mypca.load(path=os.path.join(current_path, 'data_process/pcaweights_ext.pca'))
        
    for i in range(int(input("cycles"))):
        new_data = gen_lim(1000, dataIterator)
        sampler.set_seed(i)
        print("\t\t")
        mypca = mypca.fit(new_data, PCA_COUNT, verbose=True)
    print("saving PCA...")
    mypca.save(os.path.join(current_path,'data_process/pcaweights_ext.pca'))
    print("BB!")
    
