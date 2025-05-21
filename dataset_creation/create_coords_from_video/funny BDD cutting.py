import os
import cv2
import dlib
import torch
import numpy as np
from pywavefront import Wavefront
from datetime import datetime
import time

import depth_adding.mappings as mappings  # ← mapping + ключевые точки DLIB
import depth_adding.depthFinder as depthFinder  # ← старый depthFinder
import depth_adding.imgutils as imgutils # ← нормализация, шум, визуализация
import depth_adding.coordsParser as coordsParser # ← парсер координат (если нужно)

# === Настройки ===
DLIB_DOTS = 68
TOTAL_DOTS = 72
VIDEO_PATH = 'video.mp4'
SAVE_DIR = 'output'
MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'
FACE_DETECTOR_PATH = 'mmod_human_face_detector.dat'
OBJ_MODEL_PATH = 'demoface.obj'
ADDPOINTS_PATH = 'newPoints.txt'

START_FRAME_ID = 27001

IMG_DIR = os.path.join(SAVE_DIR, "images")
COORD_DIR = os.path.join(SAVE_DIR, "coords")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(COORD_DIR, exist_ok=True)

# === Инициализация ===
detector = dlib.cnn_face_detection_model_v1(FACE_DETECTOR_PATH)
predictor = dlib.shape_predictor(MODEL_PATH)
face = Wavefront(OBJ_MODEL_PATH)
adlist = []

def get_add_list(filename: str):
    lst = []
    for line in open(filename):
        lst.append(np.array(list(map(int, line.strip().split()))))
    return lst

adlist = get_add_list(ADDPOINTS_PATH)
idmap = mappings.dlibToMeshMapping

# === Обработка видео ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = START_FRAME_ID

print("Nakonec vsyo zagruzilos', let's start!")

while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    dets = detector(frame, 1)
    if not dets:
        frame_id += 1
        continue

    for det in dets:
        face_rect = dlib.rectangle(int(det.rect.left()), int(det.rect.top()),
                                   int(det.rect.right()), int(det.rect.bottom()))
        landmarks = predictor(frame, face_rect)

        coords = torch.zeros(TOTAL_DOTS, 2)
        for i in range(DLIB_DOTS):
            coords[i][0] = landmarks.part(i).x
            coords[i][1] = landmarks.part(i).y

        # дополнительные точки
        for line in adlist:
            idx_target = line[0]
            
            idmap[idx_target] = line[1]
            
            src_indices = line[2:]
            
            coords[idx_target, 0] = np.mean([coords[i, 0].item() for i in src_indices])
            coords[idx_target, 1] = np.mean([coords[i, 1].item() for i in src_indices])

        # → глубина для всех точек с использованием старого пайплайна
        raw_coords_list = [[int(x), int(y)] for x, y in coords.tolist()]  # to int for compatibility
        depthlist = depthFinder.findDepth(OBJ_MODEL_PATH,
                                          raw_coords_list,
                                          mappings.lEyeCornerIdImgDlib,
                                          mappings.rEyeCornerIdImgDlib,
                                          mappings.upNoseIdImgDlib,
                                          idmap,
                                          accuracy=100,
                                          debug=False)

        # добавим шум (опционально)
        depthlist = imgutils.noiseAdding(depthlist, 0.02)

        # нормализация x, y
        height, width = frame.shape[:2]
        coords_np = coords.numpy()
        coords_np[:, 0] /= width
        coords_np[:, 1] /= height

        # объединяем x, y, z
        coords3d = np.column_stack((coords_np, depthlist))

        # сохранение
        img_name = os.path.join(IMG_DIR, f"dataimg{frame_id}.jpg")
        coord_name = os.path.join(COORD_DIR, f"condcords{frame_id}_3d.txt")

        cv2.imwrite(img_name, frame)

        with open(coord_name, "w") as f:
            for x, y, z in coords3d:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        print(f"[{frame_id}] Сохранено: {img_name} + {coord_name}")
        print(f'{time.time() - start_time}')

    frame_id += 1

cap.release()
print("Обработка завершена.")