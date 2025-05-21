import os
import pandas as pd
import torch
import joblib
from sklearn.decomposition import PCA
import time

VERSION = "OLD"

PCA_COUNT = 16

# class MakerPCA():
#     def __init__(self):
#         self.pca = PCA()

#     def fit(self, dataset, component_num, verbose = False):
#         df = []
#         for _, coords_batch in dataset:
#             for coords in coords_batch:
#                 coords = coords.permute(1, 0)[0:3]
#                 coords = coords.reshape(-1, coords.numel()).squeeze(0).cpu().numpy()
#                 df.append(coords)
            
#         df = pd.DataFrame(df)

#         self.pca = PCA(n_components=component_num)
#         self.pca.fit(df)

#         if verbose:
#             exvr = self.pca.explained_variance_ratio_.sum()
#             print(f'explained variance of {component_num} PCA components: {exvr}')

#         return self
    
#     def save(self, path):
#         joblib.dump(self.pca, path)

#     def load(self, path):
#         self.pca = joblib.load(path)
#         print(self.pca)
#         return self
    
#     def compress(self, coords_batch):
#         df = []
#         for coords in coords_batch:
#             coords = coords.permute(1, 0)[0:3]
#             coords = coords.reshape(-1, coords.numel()).cpu().squeeze(0).numpy()
#             df.append(coords)
#         df = pd.DataFrame(df)
#         df_trans = self.pca.transform(df)
#         return torch.Tensor(df_trans)

#     def decompress(self, PCA_batch):
#         df = []
#         for coords in PCA_batch:
#             df.append(coords)
#         df = pd.DataFrame(df)
#         df_trans = self.pca.inverse_transform(df)
#         tslist = []
#         for coords in df_trans.iterrows():  
#             tslist.append(torch.Tensor(coords[1].values.astype(float)))
#         ts = torch.stack(tslist)
#         return ts
    
#         df = pd.DataFrame([coords.tolist() for coords in PCA_batch])
#         restored = self.pca.inverse_transform(df)  # shape: (B, 216)

#         output = []
#         for row in restored:
#             pts = torch.tensor(row, dtype=torch.float32).view(-1, 3)  # (72, 3)
#             output.append(pts)

#         return torch.stack(output)  # shape: (B, 72, 3)

# i rewrite old PCA для совместимости
class MakerPCA:
    def __init__(self):
        self.pca = PCA()
        self.dots = 72  # по умолчанию

    def fit(self, dataset, verbose=False):
        """
        dataset: итератор по (x, coords_batch), где coords_batch — (B, 72, 3)
        """
        df = []
        for _, coords_batch in dataset:
            for coords in coords_batch:  # (72, 3)
                coords_np = coords.cpu().numpy().reshape(-1)
                df.append(coords_np)
        df = pd.DataFrame(df)

        self.pca = PCA(n_components=PCA_COUNT)
        self.pca.fit(df)

        if verbose:
            exvr = self.pca.explained_variance_ratio_.sum()
            print(f'explained variance of {PCA_COUNT} PCA components: {exvr}')

        self.dots = coords.shape[0]
        return self

    def save(self, path):
        joblib.dump(self.pca, path)

    def load(self, path):
        data = joblib.load(path)
        print(f"[DEBUG] Loaded type: {type(data)}")
        print(f"[DEBUG] Loaded keys: {getattr(data, 'keys', lambda: 'not a dict')()}")

        if isinstance(data, dict):
            self.pca = data['pca']
            self.dots = data.get('dots', 72)
        else:
            self.pca = data
            self.dots = 72  # по умолчанию
        return self

    def compress(self, coords_batch):
        """
        coords_batch: Tensor (batch_size, 72, 3)
        Returns: Tensor (batch_size, n_components)
        """
        # reshaped = coords_batch.view(coords_batch.shape[0], -1).cpu().numpy()
        # compressed = self.pca.transform(reshaped)
        # return torch.tensor(compressed, dtype=torch.float32)
    
        reshaped = coords_batch.view(coords_batch.shape[0], -1).cpu().numpy()
        reshaped = pd.DataFrame(reshaped)
        compressed = self.pca.transform(reshaped)
        return torch.tensor(compressed, dtype=torch.float32)

    def decompress(self, compressed_batch):
        """
        compressed_batch: Tensor (batch_size, n_components)
        Returns: Tensor (batch_size, 72, 3)
        """
        if compressed_batch.ndim == 1:
            compressed_batch = compressed_batch.unsqueeze(0)

        restored = self.pca.inverse_transform(compressed_batch.cpu().numpy())  # (B, 216)
        restored_tensor = torch.tensor(restored, dtype=torch.float32).view(-1, self.dots, 3)
        return restored_tensor
