import os
import cv2
import dlib
import torch
import numpy as np
from pywavefront import Wavefront
from datetime import datetime
import time

import numpy.typing as npt

import depth_adding.mappings as mappings  # ← mapping + ключевые точки DLIB
import depth_adding.depthFinder as depthFinder  # ← старый depthFinder
import depth_adding.imgutils as imgutils # ← нормализация, шум, визуализация
import depth_adding.coordsParser as coordsParser # ← парсер координат (если нужно)


# === Настройки ===
IMAGES = "images"
VIDEO = "video"
CHOOSE = IMAGES

IMAGE_SCALE_FOR_DLIB = 0.5 # 256 / 512

DLIB_DOTS = 68
TOTAL_DOTS = 72
VIDEO_PATH = 'video.mp4'
IMAGES_PATH = 'input_images_1/1_1'
SAVE_DIR = 'output_1_1'
MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'
FACE_DETECTOR_PATH = 'mmod_human_face_detector.dat'
OBJ_MODEL_PATH = 'demoface.obj'
ADDPOINTS_PATH = 'newPoints.txt'

START_FRAME_ID = 1

SKIP_FIRST_FRAMES = 0

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
frames_done = 0


def process_frame(frame: npt.NDArray[np.uint8], frame_id: int, scale : float = 1.0) -> None:
    img_orig = frame.copy()
    if scale != 1.0:
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    dets = detector(frame, 1)
    if not dets:
        print(f"[{frame_id}] Skipped: face is not detected")
        return False

    for det in dets:
        face_rect = dlib.rectangle(int(det.rect.left()), int(det.rect.top()),
                                   int(det.rect.right()), int(det.rect.bottom()))
        landmarks = predictor(frame, face_rect)

        coords = torch.zeros(TOTAL_DOTS, 2)
        for i in range(DLIB_DOTS):
            coords[i][0] = landmarks.part(i).x
            coords[i][1] = landmarks.part(i).y

        for line in adlist:
            idx_target = line[0]
            idmap[idx_target] = line[1]
            src_indices = line[2:]
            coords[idx_target, 0] = np.mean([coords[i, 0].item() for i in src_indices])
            coords[idx_target, 1] = np.mean([coords[i, 1].item() for i in src_indices])

        raw_coords_list = [[int(x), int(y)] for x, y in coords.tolist()]
        depthlist = depthFinder.findDepth(OBJ_MODEL_PATH,
                                          raw_coords_list,
                                          mappings.lEyeCornerIdImgDlib,
                                          mappings.rEyeCornerIdImgDlib,
                                          mappings.upNoseIdImgDlib,
                                          idmap,
                                          accuracy=100,
                                          debug=False)

        depthlist = imgutils.noiseAdding(depthlist, 0.02)

        height, width = frame.shape[:2]
        coords_np = coords.numpy()
        coords_np[:, 0] /= width
        coords_np[:, 1] /= height

        coords3d = np.column_stack((coords_np, depthlist))

        img_name = os.path.join(IMG_DIR, f"dataimg{frame_id}.jpg")
        coord_name = os.path.join(COORD_DIR, f"condcords{frame_id}_3d.txt")

        cv2.imwrite(img_name, img_orig)

        with open(coord_name, "w") as f:
            for x, y, z in coords3d:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        print(f"[{frame_id}] Saved: {img_name} + {coord_name}")
        return True

    return False


print("Nakonec vsyo zagruzilos', let's start!")

frame_id = START_FRAME_ID
frames_done = 0

# --- Обработка видео ---
if CHOOSE == VIDEO:
    cap = cv2.VideoCapture(VIDEO_PATH)
    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        if frames_done < SKIP_FIRST_FRAMES:
            frames_done += 1
            print(f"[{frame_id}, {frames_done}] Skipped: user set SKIP_FIRST_FRAMES")
            frame_id += 1
            continue

        process_frame(frame, frame_id, IMAGE_SCALE_FOR_DLIB)

        frames_done += 1
        print(f'{time.time() - start_time: .4f}s')

        frame_id += 1

    cap.release()

# --- Обработка изображений ---
if CHOOSE == IMAGES:
    # получаем список файлов изображений (расширения можно добавить)
    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    img_files = [f for f in os.listdir(IMAGES_PATH) if os.path.splitext(f)[1].lower() in valid_ext]
    img_files.sort()

    for img_file in img_files:
        img_path = os.path.join(IMAGES_PATH, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[{img_path}] Cannot read image")
            continue

        start_time = time.time()

        process_frame(frame, frame_id, IMAGE_SCALE_FOR_DLIB)

        frames_done += 1
        print(f"[{frame_id}] Processed image {img_file} in {time.time() - start_time:.4f}s")

        frame_id += 1

print("Обработка завершена.")
