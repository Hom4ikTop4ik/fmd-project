import data_loader_int8
import torch

batchsize = 25000
images, coords = data_loader_int8.load("I:/NSU/CV/tests/torch/data/train/coords",
                        "I:/NSU/CV/tests/torch/data/train/images", 
                        fromid = 0, toid = batchsize)
torch.save(images, 'dataset_images.pt')
torch.save(coords, 'dataset_coords.pt')