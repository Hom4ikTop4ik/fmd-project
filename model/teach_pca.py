import os
import time
import torch
import torch.nn as nn
import numpy as np

# from data_process import augment_gen
# from data_process import MakerPCA
from data_process import load

from data_process.convertor_pca import MakerPCA, PCA_COUNT, VERSION
from detector.blocked import MultyLayer

from data_process import BATCH_SIZE, DA, NET, POCHTI

AUG = POCHTI

QUICK = DA

def gen_lim(limit, generator):
    for i, obj in enumerate(generator):
        # if (i % 1 == 0) :
        # print(i)
        if i >= limit:
            break
        yield obj

# establishing paths and loading
current_path = os.path.dirname(os.path.abspath(__file__))
registry_path = os.path.join(current_path, 'registry')

# establishing devices and signal handler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(time.time())
print("devise is: ", device)
if __name__ == "__main__":
    dataLoader, sampler = load(
        BATCH_SIZE, 
        "./model/registry/dataset/train", 
        "images", 
        "coords", 
        device, 
        sampler_seed=0, 
        augments=AUG,
        workers=NET
    )

    dataIterator = iter(dataLoader)

    mypca = MakerPCA()
    if QUICK:
        again = 'n'
    else:
        again = input("Load old pca & train more - y? (y/n)")
    if (again in "yYнН"):
        print("load")
        mypca.load(path=os.path.join(current_path, 'data_process/pcaweights_ext.pca'))
        
    if QUICK:
        PCA_cycles = 1
    else:
        PCA_cycles = int(input("cycles: "))
    for i in range(PCA_cycles):
        new_data = gen_lim(1000, dataIterator)
        sampler.set_seed(i)
        mypca = mypca.fit(new_data, verbose=True)

    save_path_backup = f'data_process/pcaweights_ext.pca_{time.time()}'
    mypca.save(os.path.join(current_path, save_path_backup))
    print(f"saved to {save_path_backup}")
    
    if QUICK:
        save_or_no = 'y'
    else:
        save_or_no = input("Save trained PCA? (y/n)")

    if (save_or_no in "yYнН"):
        save_path_base = 'data_process/pcaweights_ext.pca'
        mypca.save(os.path.join(current_path, save_path_base))
        print(f"saved to {save_path_base}")

    print("BB!")
    print(time.time())
