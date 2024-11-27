import cv2
import os

def extract_frames(video_path, output_folder):
    # проверка существования выходной папки, создание при необходимости
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # открытие видео
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    # проверка успешности открытия
    if not cap.isOpened():
        print("Error: can not open the video")
        return
    
    # чтение кадров
    while True:
        ret, frame = cap.read()
        if not ret:  # конец видео
            break
        
        # сохранение кадра
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames to folder {output_folder}")

# пример использования
video_path = "video.mp4"  # путь к видео либо имя видео, если оно в текущей директории
output_folder = "output_frames"  # папка для сохранения кадров
extract_frames(video_path, output_folder)
