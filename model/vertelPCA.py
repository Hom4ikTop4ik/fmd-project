import cv2
import torch
import numpy as np
from data_process.convertor_pca import MakerPCA, PCA_COUNT
from data_process import load, BATCH_SIZE, USE_CPU_WHATEVER, NET

WINDOW_NAME = 'PCA Reconstruction'
IMAGE_WINDOW = 'Image'

DOTS = 72

if __name__ == "__main__":
    global image, coords, compressed, trackbar_values
    kchau = 1000

    device = 'cpu'
    if (USE_CPU_WHATEVER == NET):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mypca = MakerPCA().load('./model/data_process/pcaweights_16PCA.pca')

    dataLoader, sampler = load(BATCH_SIZE, "./model/registry/dataset/train", "images", "coords", device, sampler_seed=18)
    dataIterator = iter(dataLoader)

    image_batch, coord_batch = next(dataIterator)
    index_in_batch = 0

    def set_image_index(i):
        global image, coords, compressed, trackbar_values
        image = image_batch[i]
        coords = coord_batch[i]

        compressed = mypca.compress([coords])[0].numpy()
        trackbar_values = (compressed * kchau).astype(int)

        for j in range(PCA_COUNT):
            cv2.setTrackbarPos(f'PC{j}', WINDOW_NAME, trackbar_values[j] + 3 * kchau)

        update()

    def update(val=0):
        global image, coords, compressed, trackbar_values
        slider_vals = np.array([
            (cv2.getTrackbarPos(f'PC{i}', WINDOW_NAME) - 3 * kchau) / kchau
            for i in range(PCA_COUNT)
        ])
        
        decoded = mypca.decompress(torch.tensor([slider_vals], dtype=torch.float32)).reshape(-1, 3, DOTS).permute(0, 2, 1)[0]

        img = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8).copy()
        h, w = img.shape[:2]

        coords_np = coords.cpu().numpy()
        decoded_np = decoded.cpu().numpy()

        for i in range(coords_np.shape[0]):
            x1, y1 = int(coords_np[i][0] * w), int(coords_np[i][1] * h)
            x2, y2 = int(decoded_np[i][0] * w), int(decoded_np[i][1] * h)

            cv2.circle(img, (x1, y1), 2, (255, 255, 255), -1)
            cv2.circle(img, (x2, y2), 2, (0, 0, 255), -1)

        cv2.imshow(IMAGE_WINDOW, img)

    # Окно с трекбарами
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    for i in range(PCA_COUNT):
        cv2.createTrackbar(f'PC{i}', WINDOW_NAME, 3*kchau, 6*kchau, update)

    # Отдельное окно под изображение
    cv2.namedWindow(IMAGE_WINDOW, cv2.WINDOW_AUTOSIZE)

    set_image_index(index_in_batch)

    print("← → — переключение. 'q' — выход.")
    while True:
        key = cv2.waitKey(100)

        if key == ord('q'):
            break
        elif key == 81:  # ←
            if index_in_batch > 0:
                index_in_batch -= 1
                set_image_index(index_in_batch)
        elif key == 83:  # →
            index_in_batch += 1
            if index_in_batch >= len(image_batch):
                try:
                    image_batch, coord_batch = next(dataIterator)
                    index_in_batch = 0
                except StopIteration:
                    print("Закончились данные")
                    break
            set_image_index(index_in_batch)

    cv2.destroyAllWindows()
