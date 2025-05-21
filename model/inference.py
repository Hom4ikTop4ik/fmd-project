import torch.nn as nn
import torch
import cv2
import numpy as np
from data_process.convertor_pca import MakerPCA, PCA_COUNT, VERSION
from data_process import show_image_coords 
from detector import MultyLayer
from scipy.spatial import Delaunay
import os
import argparse
import time
import sys

def shrink_and_center(img: np.ndarray, k: float, target_size: int = 512) -> np.ndarray:
    """
    Уменьшает изображение в k раз, центрирует на фоне target_size x target_size,
    оставшееся пространство заполняется белым цветом.

    :param img: Входное изображение (H, W, 3), dtype=uint8
    :param k: Коэффициент уменьшения (например, k=2 уменьшит изображение в 2 раза)
    :param target_size: Размер итогового изображения (по умолчанию 512)
    :return: Новое изображение (target_size, target_size, 3), dtype=uint8
    """
    assert k > 0, "Коэффициент масштабирования должен быть положительным"

    h, w = img.shape[:2]
    new_w, new_h = int(w / k), int(h / k)

    # Масштабирование
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Создание белого фона
    white_bg = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Координаты для вставки
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2

    # Вставка в центр
    white_bg[top:top+new_h, left:left+new_w] = resized
    return white_bg


def get_first_delaunay(points):
    try:
        tri = Delaunay(points[:, :2])
    except Exception as e:
        print(f"delaunay error: {e}")
        return
    return tri

def delaunay_to_obj(points, first_delaune, filename="output.obj"):
    '''
    Generates an OBJ file from a Delaunay triangulation, using the first two coordinates.
    Args:
        points: A numpy array (68x3) containing the point coordinates (x, y, z).
        filename: The name of the file to write the OBJ data to.
    '''
    try:
        tri = Delaunay(points[:, :2])
    except Exception as e:
        print(f"delaunay error: {e}")
        return

    with open(filename, "w") as f:
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")
        for simplex in first_delaune.simplices:
            f.write(f"f {simplex[0] + 1} {simplex[1] + 1} {simplex[2] + 1}\n")

def visualize_delaunay(points, img, SCALE = 1, use_lines = True):
    img_width, img_height = img.shape[:2] 
    
    img_width  *= SCALE 
    img_height *= SCALE 

    points2d = points[:, :2]
    img_points = (points2d * np.array([img_width, img_height])).astype(np.int32)
    # img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    if SCALE != 1:
        img = cv2.resize(img, (img_width, img_height))

    try:
        delaunay = Delaunay(points2d)
    except Exception as e:
        print(f"Ошибка Delaunay: {e}")
        return img
    
    if use_lines:
        for simplex in delaunay.simplices:
            vertices = img_points[simplex]
            cv2.line(img, tuple(vertices[0]), tuple(vertices[1]), (0, 0, 255), 1 * SCALE)
            cv2.line(img, tuple(vertices[1]), tuple(vertices[2]), (0, 0, 255), 1 * SCALE)
            cv2.line(img, tuple(vertices[2]), tuple(vertices[0]), (0, 0, 255), 1 * SCALE)
    for i, p in enumerate(img_points):
        lev = 255 - int((points[i, 2] - 2.5) * 255) * 2 
        val = points[i, 2] - 2.5
        cv2.circle(img, tuple(p), 1 * SCALE, (lev, 0, 0), -1)
        # cv2.putText(img, f"{val:.3f}", tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img, f"{i}", tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * SCALE, (255, 255, 255), 1, cv2.LINE_AA)
    return img

head_desc1 = [
    ('linear', 512), 
    ('linear', 384), 
    ('linear', 256), 
    ('linear', 128), 
    ('linear', 512), 
    ('linear', 64), 
    ('linear', PCA_COUNT)
]

def test(videopath, outpath, verbose = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det1 = MultyLayer(device, PCA_COUNT, head_desc=head_desc1).to(device)
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    registry_path = os.path.join(current_path, 'registry')
    if VERSION == "NEW":
        file_weight_name = 'model_bns_GROUPS_3.pth'
    elif VERSION == "OLD":
        file_weight_name = 'model_bns_16PCA_60_epochs.pth'
    weight_load_path = os.path.join(registry_path, 'weights', file_weight_name)
    
    det1.load_state_dict(torch.load(weight_load_path))
    det1.eval()
    
    mypca = MakerPCA()
    if VERSION == "NEW":
        file_PCA_name = 'pcaweights_ext_GROUPS_3.pca'
    elif VERSION == "OLD":
        file_PCA_name = 'pcaweights_16PCA.pca'

    mypca.load(os.path.join(current_path, f'data_process/{file_PCA_name}'))
    

    # videopath = os.path.join(current_path, '../reconstruction/big_yula_est_chokoladku.mp4')
    cap = None
    if videopath == 'camera':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(videopath)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outpath, fourcc, 24.0, (512,512)) # имя файла, кодек, fps, размер кадра

    frameid = 0

    isfirst = True
    firsdel = None
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if verbose:
            print(frame.shape)

        if frame is None:
            break
        
        try:
            if videopath == "camera":
                frame = cv2.resize(frame, (683, 512)) # 640x480 -> 683x512
            else:
                frame = cv2.resize(frame, (512, 512))
        except Exception as e:
            break


        # frame = shrink_and_center(frame, 1.4)
        frame = torch.from_numpy(frame.astype(np.float32)).to(device) / 255
        if videopath == "camera":
            imgtens = frame[0:512, 0+85:512+85, :].permute(2, 0, 1).unsqueeze(0) # cut porovnu from left and right sides
        else:
            imgtens = frame.permute(2, 0, 1).unsqueeze(0)


        if verbose:
            print(imgtens.shape)
        
        predict = det1(imgtens)
        predict = mypca.decompress(predict)#.reshape(-1, 3, 72).permute(0, 2, 1)

        wasimg = (imgtens[0, :, :, :].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

        if verbose:
            print('shape is', predict.squeeze().cpu().shape)

        # out.write(wasimg)
        if isfirst == True:
            firsdel = get_first_delaunay(predict.squeeze())
            isfirst = False

        newimg = visualize_delaunay(predict.squeeze().cpu().numpy(), wasimg, SCALE=2)
        
        delaunay_to_obj(predict.squeeze(), firsdel, outpath + f'/facemesh{frameid:04d}.obj')

        # newimg = np.ascontiguousarray(newimg)
        # for coord in predict[0]:
        #     x, y = coord[0], coord[1]
        #     # print(coord.shape, newimg.shape, predict.shape)
        #     newimg = cv2.circle(newimg, (int(x * newimg.shape[1]), int(y * newimg.shape[0])), 2, (255, 255, 255), 2)
        
        cv2.imshow("img", newimg)
        
        frameid += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="inference of the model")

    parser.add_argument("videopath", type=str, help="path to the video file in .mp4, can be camera")
    parser.add_argument("outpath", type=str, help="path to the output folder where obj sequence will be placed")

    args = parser.parse_args()

    start_time = time.time()
    test(args.videopath, args.outpath)
    print("spent time:", time.time() - start_time)
    