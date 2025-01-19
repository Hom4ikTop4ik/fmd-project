import os
from PIL import Image
import torch
from torchvision import transforms
import modelutils

def getNumStr(coordsFileName):
    return coordsFileName[-12:-7]

def getImgName(coordsFileName):
    return "dataimg" + getNumStr(coordsFileName) + ".jpeg"

def loadIdeal(loadimage=False):
    coordsPath = "I:/NSU/CV/tests/torch/data/train/coords/condcords00131_3d.txt"
    imgPath = "I:/NSU/CV/tests/torch/data/train/images/dataimg00131.jpeg"

    if loadimage:
        img = Image.open(imgPath)
        imgtensor = transforms.ToTensor()(img)
        img.close()
    else:
        imgtensor = None

    coordsFile = open(coordsPath, "r")
    all_coords = []
    for line in coordsFile:
        coords_str = line.strip()
        coords_list = [float(x) for x in coords_str.split()]
        coords_list = coords_list[:-1]
        all_coords.append(coords_list)
    coords_tensor = torch.tensor(all_coords, dtype=torch.float32)
    coordsFile.close()
    
    if coords_tensor.shape[0] != 68:
        print("coords file", coordsPath, "is not 68 points")
    return (imgtensor, coords_tensor) if loadimage else coords_tensor

class Transform2d:
    def __init__(self, rotate_angle, pivot, scale_factor, translate):
        self.rotate_angle = rotate_angle
        self.pivot = pivot
        self.scale_factor = scale_factor
        self.translate = translate
    def __call__(self, v):
        v = v - self.pivot
        v = torch.tensor([[torch.cos(self.rotate_angle), -torch.sin(self.rotate_angle)],
                          [torch.sin(self.rotate_angle), torch.cos(self.rotate_angle)]], dtype=torch.float32) @ v
        v = v * self.scale_factor
        v = v - self.translate
        v = v + self.pivot
        return v

def find_transform(eye_L, eye_R, eye_L_pointlist, eye_R_pointlist, coords_tensor):
    # Get target coordinates from ideal face
    x_l, y_l = modelutils.get_mean_coords(eye_L_pointlist, coords_tensor)
    x_r, y_r = modelutils.get_mean_coords(eye_R_pointlist, coords_tensor)
    
    # Create source and target vectors
    x1_l, y1_l = eye_L[0], eye_L[1]
    x1_r, y1_r = eye_R[0], eye_R[1]
    
    eyelen1 = torch.sqrt((y1_r - y1_l)**2 + (x1_r - x1_l)**2)
    eyelen = torch.sqrt((y_l - y_r)**2 + (x_l - x_r)**2)
    
    angle = torch.atan((y1_r - y1_l) / (x1_r - x1_l))
    pivot = torch.tensor([(x_l + x_r) / 2, (y_l + y_r) / 2], dtype=torch.float32)
    scale_factor = eyelen1 / eyelen
    translate = torch.tensor([(x_l + x_r) / 2 - (x1_l + x1_r) / 2, (y_l + y_r) / 2 - (y1_l + y1_r) / 2], dtype=torch.float32)
    transform = Transform2d(angle, pivot, scale_factor, translate)
    return transform


# Eye landmark indices
# eye_L_pointlist = [62, 65, 0, 13, 34, 64]
# eye_R_pointlist = [16, 36, 54, 55, 53, 63]

def apply_face(eye_L, eye_R, eye_L_pointlist, eye_R_pointlist, ideal_face_tensor):
    coords_tensor = ideal_face_tensor
    transform = find_transform(eye_L, eye_R, eye_L_pointlist, eye_R_pointlist, coords_tensor)
    newcoords_tensor = torch.zeros(coords_tensor.shape, dtype=torch.float32)
    for i in range(coords_tensor.shape[0]):
        newcoords_tensor[i] = transform(coords_tensor[i])
    return newcoords_tensor


# imgtensor, coords_tensor = loadIdeal(loadimage=True)

# eye_L = torch.tensor([0.6, 0.59], dtype=torch.float32)
# eye_R = torch.tensor([0.47, 0.5], dtype=torch.float32)

# newcoords = apply_face(eye_L, eye_R, eye_L_pointlist, eye_R_pointlist)
# newcoords[0] = eye_L
# newcoords[1] = eye_R
# print(newcoords.shape, imgtensor.shape)
# xshow = modelutils.show_tensor(imgtensor, newcoords, no_z = True)
# xshow.show()
