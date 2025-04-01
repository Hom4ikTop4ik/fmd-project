import cv2

import data_loader_int8
import torch

batchsize = 25000
images, coords = data_loader_int8.load("I:/NSU/CV/tests/torch/data/train/coords",
                        "I:/NSU/CV/tests/torch/data/train/images", 
                        fromid = 0, toid = batchsize)

# List of redused images
reduced_images = []

# Reduce images by using OpenCV
for i in range(len(images)):
    img = images[i].numpy().transpose(1, 2, 0) # Преобразование тензора в изображение: каналы цвета на третье место
    img = cv2.resize(img, (512, 512))
    reduced_images.append(torch.tensor(img.transpose(2, 0, 1))) # transpose back and add to list

torch.save(reduced_images, 'dataset_images.pt')
torch.save(coords, 'dataset_coords.pt')
