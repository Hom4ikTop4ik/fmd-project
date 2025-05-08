import os
# from .data_loader import load
import pandas as pd
import torch
import joblib
from sklearn.decomposition import PCA
import time

class MakerPCA():
    def __init__(self):
        self.pca = PCA()

    def fit(self, dataset, component_num, verbose = False):
        df = []
        for batch in dataset:
            for coords in batch[1]:
                coords = coords.permute(1, 0)[0:3]
                coords = coords.reshape(-1, coords.numel()).squeeze(0).cpu().numpy()
                df.append(coords)
            
        df = pd.DataFrame(df)

        self.pca = PCA(n_components=component_num)
        self.pca.fit(df)

        if verbose:
            exvr = self.pca.explained_variance_ratio_.sum()
            print(f'explained variance of {component_num} PCA components: {exvr}')

        return self
    
    def save(self, path):
        joblib.dump(self.pca, path)

    def load(self, path):
        self.pca = joblib.load(path)
        return self

    
    def compress(self, coords_batch):
        df = []
        for coords in coords_batch:
            coords = coords.permute(1, 0)[0:3]
            coords = coords.reshape(-1, coords.numel()).cpu().squeeze(0).numpy()
            df.append(coords)
        df = pd.DataFrame(df)
        df_trans = self.pca.transform(df)
        return torch.Tensor(df_trans)

    def decompress(self, PCA_batch):
        df = []
        for coords in PCA_batch:
            df.append(coords)
        df = pd.DataFrame(df)
        df_trans = self.pca.inverse_transform(df)
        tslist = []
        for coords in df_trans.iterrows():  
            tslist.append(torch.Tensor(coords[1].values.astype(float)))
        ts = torch.stack(tslist)
        return ts


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.dirname(__file__))
    registry_path = os.path.join(current_path, 'registry')
    dataset = load(500, 40, os.path.join(current_path, registry_path, 'dataset'))
    print('loaded')
    convertor = MakerPCA().fit(dataset, 40, verbose=True)
    print('fitted')
    # convertor = MakerPCA().load('pcaweights.pca')
    
    coordbatch = dataset[0][1]
    print('coordsbatch', coordbatch.shape)
    start = time.time()
    compressed = convertor.compress(coordbatch)
    end = time.time()
    print(f'compressed in {end - start:.3f} seconds: {compressed.shape}')
    print(compressed)
    start = time.time()
    decompressed = convertor.decompress(compressed)
    end = time.time()
    print(f'decompressed in {end - start:.3f} seconds: {decompressed.shape}')
    print(decompressed)
