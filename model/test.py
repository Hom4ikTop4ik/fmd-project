import torch
import os
from data_process import load

import torch
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_landmarks(landmarks, image_size=(100, 100)):
    if isinstance(landmarks, torch.Tensor): landmarks = landmarks.cpu().numpy()
    height, width = image_size
    plt.figure(figsize=(width/100, height/100), dpi=100)
    plt.xlim(0, width)
    plt.ylim(height, 0)
    plt.axis('off')
    for i, coords in enumerate(landmarks):
        x = coords[0]
        y = coords[1]
        x_pixel = int(x * width)
        y_pixel = int(y * height)
        plt.scatter(x_pixel, y_pixel, marker='o', color='red', s=10)
        plt.text(x_pixel + 2, y_pixel + 2, str(i), color='black', fontsize=8)
    plt.tight_layout(pad=0)
    plt.show()



current_path = os.path.dirname(os.path.abspath(__file__))
registry_path = os.path.join(current_path, 'registry')
weight_save_path = os.path.join(registry_path, 'weights', 'lvl1det_bns.pth')
dataset = load(500, 40, os.path.join(current_path, registry_path, 'dataset'))

plot_landmarks(dataset[0][1][0])
