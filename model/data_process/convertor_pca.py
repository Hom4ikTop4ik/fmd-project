import os
import pandas as pd
import torch
import numpy as np
import joblib
from sklearn.decomposition import PCA

VERSION = "NEW"

CAN_TRAIN_MORE = True

# [from, to] включительно
# PCA_COUNT = sum(PCi_COUNT)
# (from, to, PCi_COUNT)
PCA_LIST_0 = [
    (0, 71, 16)
]

PCA_LIST_1 = [
    (0, 16, 16), # челюсть

    (17, 21, 8), # left brov'
    (22, 26, 8), # right brov' 

    (27, 35, 8), # nose
    
    (36, 41, 16), # left eye
    (42, 47, 16), # right eye

    (48, 59, 16), # mouth 1
    (60, 67, 16), # mouth 2

    (68, 71, 8) # 4 added points
]

PCA_LIST_2 = [(a, b, c//2) for (a, b, c) in PCA_LIST_1]

PCA_LIST_3 = []
for (aa,bb,cc) in PCA_LIST_1:
    a = aa
    b = bb
    if cc == 16:
        c = 6
    elif cc == 8:
        c = 4
    PCA_LIST_3.append((a,b,c))

PCA_LIST_4 = [
    (0, 16, 8),

    (17, 21, 6),
    (22, 26, 6), 

    (27, 35, 6),
    
    (36, 41, 8),
    (42, 47, 8),

    (48, 59, 8),
    (60, 67, 8),

    (68, 71, 6)
]

PCA_LIST_LIST_0 = [
    # 0
    (range(0, 16+1), 8), # челюсть

    # 1. 2
    (range(17, 21+1), 6), # left brov'
    (range(22, 26+1), 6), # right brov' 

    # 3. 4
    (range(27, 30+1), 6), # nose_vertical
    (range(31, 35+1), 6), # nose_horizontal
    
    # 5. 6. 7. 8
    ([39, 38, 37], 4), # left eye up
    ([36, 41, 40], 4), # left eye down
    ([42, 43, 44], 4), # right eye up
    ([45, 46, 47], 4), # right eye down

    # 9. 10. 11
    # (range(48, 59+1), 6), # mouth 1
    # (range(60, 67+1), 6), # mouth 2
    ([48, 54, 60, 64], 6), # уголки рта, оранжевое
    ([49, 50, 51, 52, 53, 61, 62, 63], 6), # верхняя часть рта
    ([55, 56, 57, 58, 59, 65, 66, 67], 6), # нижняя часть рта
    
    # 12
    (range(68, 71+1), 6) # 4 added points
]

# 6 много для вертикали носа
PCA_LIST_LIST_1 = [
    # 0
    (range(0, 16+1), 8), # челюсть

    # 1. 2
    (range(17, 21+1), 6), # left brov'
    (range(22, 26+1), 6), # right brov' 

    # 3. 4
    (range(27, 30+1), 4), # nose_vertical
    (range(31, 35+1), 6), # nose_horizontal
    
    # 5. 6. 7
    ([36, 39], 4), # left eye corners
    ([37, 38], 4), # left eye up
    ([40, 41], 4), # left eye down
    
    # 8. 9. 10
    ([42, 45], 4), # right eye corners
    ([43, 44], 4), # right eye up
    ([46, 47], 4), # right eye down

    # 11. 12. 13
    # (range(48, 59+1), 6), # mouth 1
    # (range(60, 67+1), 6), # mouth 2
    ([48, 54, 60, 64], 6), # уголки рта, оранжевое
    ([49, 50, 51, 52, 53, 61, 62, 63], 6), # верхняя часть рта
    ([55, 56, 57, 58, 59, 65, 66, 67], 6), # нижняя часть рта
    
    # 14
    (range(68, 71+1), 6) # 4 added points
]

PCA_LIST = PCA_LIST_LIST_1

PERESTANOVKA = [] # хороший анекдот
for i, _ in PCA_LIST:
    PERESTANOVKA += i

def permute(our_decompressed_points : torch.Tensor):
    true = our_decompressed_points.clone()
    idx = 0
    for true_idx in PERESTANOVKA:
        true[true_idx] = our_decompressed_points[idx]
        idx += 1
    return true

PCA_COUNT = sum([PCi_count for (_, PCi_count) in PCA_LIST])

# class MakerMultyPCA:
class MakerPCA:
    def __init__(self, pca_list=PCA_LIST):
        """
        pca_list: список троек (from_idx, to_idx, n_components) включительно
        """
        print("\t__init__ PCA")
        self.pca_list = pca_list
        self.pcas = [PCA(n_components=n_comp) for _, n_comp in self.pca_list]
    
    def _check_initialized(self):
        if (not self.pcas) or any(pca is None for pca in self.pcas):
            raise RuntimeError("GroupedPCA: PCA-модели не обучены или не загружены. Сначала вызовите fit() или load().")

    def fit(self, dataset, verbose=False):
        """
        dataset: итерируемый объект, возвращающий батчи координат (batch_size, 72, 3)
        """
        num_groups = len(self.pca_list)
        groups_data = [[] for _ in range(num_groups)]

        for _, coords_batch in dataset:
            for coords in coords_batch:  # coords: (72, 3)
                coords_np = coords.cpu().numpy()  # shape: (72, 3)
                for i, (indices, _) in enumerate(self.pca_list):
                    group = coords_np[indices].reshape(-1)  # (count*3,)
                    groups_data[i].append(group)

        for i, (_, _) in enumerate(self.pca_list):
            if not CAN_TRAIN_MORE:
                if self.is_fitted(self.pcas[i]):
                    if verbose:
                        print(f"PCA group {i} already fitted, skipping")
                    continue
            
            pca = self.pcas[i]
            data = np.stack(groups_data[i], axis=0)
            pca.fit(data)
            if verbose:
                evr = pca.explained_variance_ratio_.sum()
                print(f'Group {i}: [{self.pca_list[i][0]}:{self.pca_list[i][1]}] -> {pca.n_components_} components, explained variance = {evr:.4f}')

        return self

    def compress(self, coords_batch):
        """
        coords_batch: Tensor (batch_size, 72, 3)
        Возвращает: Tensor (batch_size, sum(n_components))
        """
        self._check_initialized()

        batch_out = []
        for coords in coords_batch:  # (72, 3)
            coords_np = coords.cpu().numpy()
            compressed = []
            for (indices, _), pca in zip(self.pca_list, self.pcas):
                group = coords_np[indices].reshape(1, -1)  # (1, count*3)
                comp = pca.transform(group)[0]  # (n_components,)
                compressed.append(torch.tensor(comp, dtype=torch.float32))
            batch_out.append(torch.cat(compressed))
        return torch.stack(batch_out)

    def decompress(self, compressed_batch):
        """
        compressed_batch: Tensor (batch_size, sum(n_components))
        Возвращает: Tensor (batch_size, 72, 3)
        """
        self._check_initialized()

        output = []
        for row in compressed_batch:  # (sum(n_components),)
            pointer = 0
            points = []
            
            for (_, n_comp), pca in zip(self.pca_list, self.pcas):
                part = row[pointer : pointer+n_comp].detach().cpu().numpy().reshape(1, -1)

                restored = pca.inverse_transform(part)[0]  # shape: (group_len,)
                pts = torch.tensor(restored, dtype=torch.float32).view(-1, 3)  # (count, 3)

                points.append(pts)
                pointer += n_comp


            coords = torch.cat(points, dim=0)  # (72, 3)
            true_coords = permute(coords)
            output.append(true_coords)
        return torch.stack(output)

    def save(self, path):
        obj = {'pca_list': self.pca_list, 'pcas': self.pcas}
        joblib.dump(obj, path)

    def load(self, path, verbose = False):
        obj = joblib.load(path)
        self.pca_list = obj['pca_list']
        self.pcas = obj['pcas']

        if verbose:
            print(f'Loaded {len(self.pcas)} PCA groups:')
            print(f'List:')
            print(f'{self.pca_list}')
            print(f'{self.pcas}')

        return self
    
    # old mode
    def load_old(self, path):
        self.pcas = [joblib.load(path)]
        print(self.pcas)
        return self
