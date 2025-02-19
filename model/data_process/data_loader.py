import torch
import os

def load(size: int, bsize: int,
         dataset_path: str = '.',
         imagesfile: str = 'dataset_images.pt',
         coordsfile: str = 'dataset_coords.pt'):
    '''
    size - count of batches. 
    bsize - size of the batch.
    dataset_path - path to the folder with two tensors, images and coords.
    coordsfile - name of the file of coords tensor.
    imagssfile - name of the file of images tensor.
    '''
    images = torch.load(os.path.join(dataset_path, imagesfile))
    coords = torch.load(os.path.join(dataset_path, coordsfile))

    dataset = []
    for batch_id in range(0, size):
        print(f'{0 + batch_id * bsize} - {bsize-1 + batch_id * bsize}')
        imagesbatch = [images[i + batch_id * bsize] for i in range(0, bsize)]
        coordsbatch = [coords[i + batch_id * bsize] for i in range(0, bsize)]
        dataset.append((torch.stack(imagesbatch), torch.stack(coordsbatch)))
    return dataset
