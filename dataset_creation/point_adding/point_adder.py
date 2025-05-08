from data_loader import load_coords
from depth_finder import find_depth
import numpy as np
import os
import torch
from pywavefront import Wavefront

def get_add_list(filename: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    lst = []
    for line in open(path):
        lst.append(np.array(list(map(int, line.split()))))
    return lst


current_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
registry_path = os.path.join(current_path, 'model/registry')
weight_save_path = os.path.join(registry_path, 'weights', 'model_bns.pth')
coords = load_coords(200000, os.path.join(current_path, registry_path, 'dataset'), 
                     coordsfile='dataset_coords_68.pt')

adlist = get_add_list('data.txt')

extended_coords_list = []
clen = len(coords)
for i, coord in enumerate(coords):
    excoord = torch.zeros(68 + len(adlist), 3)
    excoord[:68] = coord
    for line in adlist:
        xmean = np.array([coord[i][0] for i in line[2:]]).mean()
        ymean = np.array([coord[i][1] for i in line[2:]]).mean()
        excoord[line[0], 0] = float(xmean)
        excoord[line[0], 1] = float(ymean)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demoface.obj')
    face = Wavefront(path)
    depths = find_depth(face, excoord,
                          1107, 13,
                          1967, 52,
                          2104, 5, 
                          [line[1] for line in adlist],
                          accuracy=100,debug=False)
    tsd = torch.Tensor(depths)
    excoord[68:, 2] = tsd 
    extended_coords_list.append(excoord)
    print(f'iteration: {i} of {clen}')


print('completed, saving')

extended_coords_tensor = torch.stack(extended_coords_list)
torch.save(extended_coords_tensor, os.path.join(registry_path, 'extended_coords.pt'))
