import cv2
import torch
import numpy as np
from data_process.convertor_pca import MakerPCA, PCA_COUNT, PCA_LIST
from data_process import load, BATCH_SIZE, USE_CPU_WHATEVER, DA, NET, show_image_coords
import sys
import time

DEBUG = NET

# Указываем индекс группы, чьи компоненты будут отображены трекбарами
ACTIVE_PCA_GROUP_INDEX = 0  # Например: 0 — челюсть, 1 — левая бровь и т.д.

WINDOW_NAME = 'PCA Reconstruction'
IMAGE_WINDOW = 'Image'

DOTS = 72

global image, coords, trackbar_values, cnt, cnt2
global compressed
cnt = 0
cnt2 = 0
compressed = None
if __name__ == "__main__":
    kchau = 1000

    device = 'cpu'
    if (USE_CPU_WHATEVER == NET):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mypca = MakerPCA().load('./model/data_process/pcaweights_ext_LIST_0.pca')

    dataLoader, sampler = load(BATCH_SIZE, "./model/registry/dataset/train", "images", "coords", 
                               device, sampler_seed=123, augments=NET, workers=NET)
    dataIterator = iter(dataLoader)

    image_batch, coord_batch = next(dataIterator)
    index_in_batch = 0

    def set_image_index(i):
        global image, coords, trackbar_values, cnt
        global compressed
        
        image = image_batch[i]
        coords = coord_batch[i]

        compressed = mypca.compress([coords])
        
        compressed = compressed[0].numpy()

        trackbar_values = (compressed * 1*kchau).astype(int)
        for j in range(active_group_comp_count):
            comp_val = trackbar_values[active_trackbar_offset + j]
            
            cv2.setTrackbarPos(f'PC{j}', WINDOW_NAME, comp_val + 3*kchau)
        
        slider_vals = compressed.copy()
        
        for i in range(active_group_comp_count):
            val = (cv2.getTrackbarPos(f'PC{i}', WINDOW_NAME) - 3*kchau) / (1.0*kchau)
            slider_vals[active_trackbar_offset + i] = val

        if DEBUG:
            dec = mypca.decompress(torch.tensor([slider_vals], dtype=torch.float32))
            show_image_coords(image, dec[0])

        update()

    def update(val = 0):
        global image, coords, trackbar_values, cnt, cnt2
        global compressed

        if compressed is None:
            cnt += 1
            if DEBUG:
                print(f'\t\tcnt: {cnt}')
            return
        if type(compressed) == type(0):
            cnt2 += 1
            if DEBUG:
                print(f'\t\t\tcnt2: {cnt2}')
            return
        
        # Начинаем с оригинального compressed-вектора
        slider_vals = compressed.copy()

        # Обновляем только компоненты выбранной группы
        for i in range(active_group_comp_count):
            val = (cv2.getTrackbarPos(f'PC{i}', WINDOW_NAME) - 3*kchau) / (1.0*kchau)
            slider_vals[active_trackbar_offset + i] = val

        decoded = mypca.decompress(torch.tensor([slider_vals], dtype=torch.float32))
        # decoded = mypca.decompress(torch.tensor([compressed], dtype=torch.float32))
        decoded = decoded[0]

        img = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8).copy()
        h, w = img.shape[:2]

        coords_np = coords.cpu().numpy()
        decoded_np = decoded.cpu().numpy()

        # for i in range(coords_np.shape[0]):
        for i in indices:
            x1, y1 = int(coords_np[i][0] * w), int(coords_np[i][1] * h)
            x2, y2 = int(decoded_np[i][0] * w), int(decoded_np[i][1] * h)

            cv2.circle(img, (x1, y1), 3, (255, 255, 255), -1)
            cv2.circle(img, (x2, y2), 3, (0, 0, 255), -1)

        cv2.imshow(IMAGE_WINDOW, img)

    # Окно с трекбарами
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    indices, active_group_comp_count = PCA_LIST[ACTIVE_PCA_GROUP_INDEX]
    active_trackbar_offset = sum([n for _, n in PCA_LIST[:ACTIVE_PCA_GROUP_INDEX]])

    for i in range(active_group_comp_count):
        cv2.createTrackbar(f'PC{i}', WINDOW_NAME, 3*kchau, 2*3*kchau, update)

    # Отдельное окно под изображение
    cv2.namedWindow(IMAGE_WINDOW, cv2.WINDOW_AUTOSIZE)

    set_image_index(index_in_batch)

    print("f← →g — переключение. 'q' — выход.")
    while True:
        key = cv2.waitKey(100)

        if key == ord('q'):
            break
        elif key == ord('f'):  # ←
            index_in_batch = (index_in_batch - 1) % len(image_batch)
            set_image_index(index_in_batch)
                
        elif key == ord('g'):  # →
            index_in_batch = (index_in_batch + 1) % len(image_batch)
            set_image_index(index_in_batch)

    cv2.destroyAllWindows()
