import os
import datetime
import torch
import torch.nn as nn
import data_process.utils as utils

from data_process import augment_gen
from data_process import load
from cascade_detector import Level1Detector

current_path = os.path.dirname(os.path.abspath(__file__))
dataset = load(1000, 20, os.path.join(current_path, 'registry/dataset'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("devise is: ", device)

eye_L_pointlist = [62, 65, 0, 13, 34, 64]
eye_R_pointlist = [16, 36, 54, 55, 53, 63]


lvl1det = Level1Detector(device).to(device)
optimizer = torch.optim.Adam(lvl1det.parameters(), 0.001)
criterion = nn.MSELoss().to(device)
iteration = 0
for bt_images, bt_coords in augment_gen(dataset, epochs=2, noise=0.02):
    iteration += 1
    print(1, datetime.datetime.now().time())
    truth = torch.stack([
        utils.get_mean_coords(eye_L_pointlist, bt_coords),
        utils.get_mean_coords(eye_R_pointlist, bt_coords)
    ]).permute(2, 0, 1).to(device)
    bt_images = bt_images.to(device)
    print(2, datetime.datetime.now().time())
    ans = lvl1det(bt_images)
    print(3, datetime.datetime.now().time())
    loss = criterion(ans, truth)
    print(4, datetime.datetime.now().time())
    optimizer.zero_grad()
    print(5, datetime.datetime.now().time())
    loss.backward()
    print(6, datetime.datetime.now().time())
    optimizer.step()
    print(7, datetime.datetime.now().time())
    if iteration % 1 == 0:
        print(f'loss {loss:.5f}, iteration: {iteration}')
    
