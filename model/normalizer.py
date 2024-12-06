import os
import random
import torch

class Scaler():
    def __init__(self):
        self.min = 0 
        self.max = 1
    def fit(self, data):
        self.min = data.min()
        self.max = data.max()
    def transform(self, data):
        return (data - self.min) / (self.max - self.min)
    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min

class MinMaxNormalizer:
    def __init__(self):
        self.min = None
        self.max = None
    
    def fit(self, tensors):
        # Find min and max of z-coordinates
        all_z = torch.cat([tensor[:, 2] for tensor in tensors])
        self.min = all_z.min()
        self.max = all_z.max()
        print(f"Fitted normalizer: min={self.min:.4f}, max={self.max:.4f}")
    
    def transform(self, tensor):
        if self.min is None:
            raise ValueError("Call fit() first")
            
        if len(tensor.shape) == 2:  # Single tensor [68, 3]
            tensor[:, 2] = (tensor[:, 2] - self.min) / (self.max - self.min)
        else:  # Batch [batch_size, 68, 3]
            tensor[:, :, 2] = (tensor[:, :, 2] - self.min) / (self.max - self.min)
        return tensor
    
    def inverse_transform(self, tensor):
        if self.min is None:
            raise ValueError("Call fit() first")
            
        if len(tensor.shape) == 2:  # Single tensor [68, 3]
            tensor[:, 2] = tensor[:, 2] * (self.max - self.min) + self.min
        else:  # Batch [batch_size, 68, 3]
            tensor[:, :, 2] = tensor[:, :, 2] * (self.max - self.min) + self.min
        return tensor