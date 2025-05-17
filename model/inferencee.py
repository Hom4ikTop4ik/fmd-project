import torch.nn as nn
import torch
import cv2
import numpy as np
from data_process import MakerPCA
from detector import MultyLayer
from scipy.spatial import Delaunay
import os
import argparse
import random
from time import sleep as rf


def get_first_delaunay(points):
    try:
        tri = Delaunay(points[:, :2])
    except Exception as e:
        print(f"delaunay error: {e}")
        return
    return tri

def detranslate(coordscur: torch.Tensor, coordspred: torch.Tensor):
    if coordspred == None:
        return coordscur
    move_x = torch.mean(coordscur[0, :, 0]) - 0.5
    move_y = torch.mean(coordscur[0, :, 1]) - 0.5

    print(f"detranslate by {move_x} on x and {move_y} on y")
    
    coordsans = coordscur.clone()
    coordsans -= torch.Tensor([move_x, move_y, 0])
    return coordsans

def derotate(coordscur: torch.Tensor, coordspred: torch.Tensor):
    if coordspred == None:
        return coordscur
    # [[x[...], y[...], z[...]]]
    
    mid_x = torch.mean(coordscur[0, :, 0])
    mid_y = torch.mean(coordscur[0, :, 1])
    
    coordscur[0, :, 0] -= mid_x
    coordscur[0, :, 1] -= mid_y

    nom = coordspred[0, :, 0] * coordscur[0, :, 0] + coordspred[0, :, 1] * coordscur[0, :, 1]
    denom = torch.sqrt(coordspred[0, :, 0] ** 2 + coordspred[0, :, 1] ** 2) * torch.sqrt(coordscur[0, :, 0] ** 2 + coordscur[0, :, 1] ** 2)
    toarcos = nom / denom
    print('toarcos: ', toarcos)
    angle = torch.acos(toarcos)
    print('anglenotmean: ', angle)
    
    angle = angle.mean()
    print('angle', angle)
    for i in range(coordscur.shape[1]):
        coordscur[0][i][:2] = torch.matmul(torch.Tensor([[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]]), coordscur[0][i][:2])
    coordscur[0, :, 0] += mid_x
    coordscur[0, :, 1] += mid_y
    print("derotate", coordscur.shape)
    return coordscur



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

def visualize_delaunay(points, img, lines = False):
    img_width, img_height = 512, 512
    points2d = points[:, :2]
    img_points = (points2d * np.array([img_width, img_height])).astype(np.int32)
    # img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    try:
        delaunay = Delaunay(points2d)
    except Exception as e:
        print(f"Ошибка Delaunay: {e}")
        return img
    if lines:
        for simplex in delaunay.simplices:
            vertices = img_points[simplex]
            cv2.line(img, tuple(vertices[0]), tuple(vertices[1]), (0, 0, 255), 1)
            cv2.line(img, tuple(vertices[1]), tuple(vertices[2]), (0, 0, 255), 1)
            cv2.line(img, tuple(vertices[2]), tuple(vertices[0]), (0, 0, 255), 1)
    for i, p in enumerate(img_points):
        lev = 255 - int((points[i, 2] - 2.5) * 255) * 2 
        val = points[i, 2] - 2.5
        cv2.circle(img, tuple(p), 1, (lev, 0, 0), -1)
        # cv2.putText(img, f"{val:.3f}", tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{i}", tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def test(videopath, outpath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    det1 = MultyLayer(device).to(device)
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    registry_path = os.path.join(current_path, 'registry')
    weight_load_path = os.path.join(registry_path, 'weights', 'model_bns_gray1.pth')
    
    # det1.load_state_dict(torch.load(weight_load_path))
    det1.load_state_dict(torch.load(weight_load_path, map_location=torch.device('cpu')))
    det1.eval()
    
    mypca = MakerPCA()
    mypca.load(os.path.join(current_path,'data_process/pcaweights_ext.pca'))
    

    # videopath = os.path.join(current_path, '../reconstruction/big_yula_est_chokoladku.mp4')
    cap = None
    if videopath == '^_^':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(videopath)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outpath, fourcc, 15.0, (432,432)) #имя файла, кодек, fps, размер кадра

    frameid = 0

    isfirst = True
    firsdel = None

    lastpred = None 
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        print('size of frame is', frame.shape)
        frame = frame[:, 100:580, :]
        frame = cv2.resize(frame, (512, 512))
        # frame = cv2.resize(frame, (720, 432))
        frame = torch.from_numpy(frame.astype(np.float32)).to(device) / 255
        imgtens = frame.permute(2, 0, 1)[:, 0:512, 0:512].unsqueeze(0)
        print(imgtens.shape)
        predict = det1(imgtens)
        predict = mypca.decompress(predict).reshape(-1, 3, 72).permute(0, 2, 1)

        wasimg = (imgtens[0, :, :, :].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

        print('shape is', predict.squeeze().cpu().shape)

        # out.write(wasimg)
        if isfirst == True:
            firsdel = get_first_delaunay(predict.squeeze())
            isfirst = False

        # predict = detranslate(predict, lastpred)
        # predict = derotate(predict, lastpred)
        lastpred = predict

        newimg = visualize_delaunay(predict.squeeze().cpu().numpy(), wasimg)
        
        delaunay_to_obj(predict.squeeze(), firsdel, outpath + f'/facemesh{frameid:04d}.obj')

        newimg = np.ascontiguousarray(newimg)
        for coord in predict[0]:
            x, y = coord[0], coord[1]
            x1, y1 = x, y
            newX, newY = int(x1 * newimg.shape[1]), int(y1 * newimg.shape[0])
            # print(coord.shape, newimg.shape, predict.shape)
            newimg = cv2.circle(newimg, (newX, newY), 2, (255, 255, 255), 2)
            print(f"{newX: 4d} {newY : 4d} {coord[2].item() : 2.4f}")
        print()
        
        name = "img"
        cv2.imshow(name, newimg)
        
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
    test(args.videopath, args.outpath)
    