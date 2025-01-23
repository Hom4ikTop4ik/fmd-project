import torch
import os

def load(size: int, bsize: int,
         dataset_path: str = '.',
         coordsfile: str = 'dataset_images.pt',
         imagesfile: str = 'dataset_coords.pt'):
    '''
    size - count of batches. 
    bsize - size of the batch.
    '''
    images = torch.load(os.path.join(dataset_path, coordsfile))
    coords = torch.load(os.path.join(dataset_path, imagesfile))
    dataset = []
    for batch_id in range(0, size):
        imagesbatch = [images[i + batch_id * bsize] for i in range(0, bsize)]
        coordsbatch = [coords[i + batch_id * bsize] for i in range(0, bsize)]
        dataset.append((torch.stack(imagesbatch), torch.stack(coordsbatch)))
    return dataset