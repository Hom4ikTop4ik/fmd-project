from data_process import augment_gen
from data_process import load
import os

current_path = os.path.dirname(os.path.abspath(__file__))
dataset = load(1000, 20, os.path.join(current_path, 'registry/dataset'))

for bt_images, bt_coords in augment_gen(dataset, epochs=2):
    
