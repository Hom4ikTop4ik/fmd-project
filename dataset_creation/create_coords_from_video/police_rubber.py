import os
import re
import sys

# === Константы по умолчанию ===
DEFAULT_MODE = 'missing_coords'  # или 'missing_images'
DEFAULT_IMAGES_PATH = './output/images'
DEFAULT_COORDS_PATH = './output/coords'

def check_missing_coords(images_path: str, coords_path: str):
    pattern = re.compile(r'dataimg(\d+)\.jpg')
    missing_coords = []

    for filename in os.listdir(images_path):
        match = pattern.fullmatch(filename)
        if not match:
            continue

        frame_id = match.group(1)
        coord_filename = f"condcords{frame_id}_3d.txt"
        coord_path = os.path.join(coords_path, coord_filename)

        if not os.path.isfile(coord_path):
            missing_coords.append(filename)

    if missing_coords:
        print("Найдены улики! Следующим изображениям не хватает координатных файлов:")
        for img_file in missing_coords:
            print(" -", img_file)
    else:
        print("Всё чисто! Все изображения имеют соответствующие координатные файлы.")

def check_missing_images(images_path: str, coords_path: str):
    pattern = re.compile(r'condcords(\d+)_3d\.txt')
    missing_images = []

    for filename in os.listdir(coords_path):
        match = pattern.fullmatch(filename)
        if not match:
            continue

        frame_id = match.group(1)
        image_filename = f"dataimg{frame_id}.jpg"
        image_path = os.path.join(images_path, image_filename)

        if not os.path.isfile(image_path):
            missing_images.append(filename)

    if missing_images:
        print("Шухер! Следующим координатным файлам не хватает изображений:")
        for coord_file in missing_images:
            print(" -", coord_file)
    else:
        print("Все улики спрятаны! Все координатные файлы имеют соответствующие изображения.")

if __name__ == "__main__":
    # Обработка аргументов командной строки
    args = sys.argv[1:]

    if len(args) == 0:
        mode = DEFAULT_MODE
        images_dir = DEFAULT_IMAGES_PATH
        coords_dir = DEFAULT_COORDS_PATH
    elif len(args) == 1:
        mode = args[0]
        images_dir = DEFAULT_IMAGES_PATH
        coords_dir = DEFAULT_COORDS_PATH
    elif len(args) == 2:
        mode = args[0]
        images_dir = args[1]
        coords_dir = DEFAULT_COORDS_PATH
    elif len(args) == 3:
        mode = args[0]
        images_dir = args[1]
        coords_dir = args[2]
    else:
        print("Использование: python script.py [mode] [images_path] [coords_path]")
        print("Режимы: missing_coords, missing_images")
        sys.exit(1)

    police_modes = ['police', 'both']
    rubber_modes = ['rubber', 'both']

    if mode in police_modes:
        check_missing_coords(images_dir, coords_dir)
    
    if mode in rubber_modes:
        check_missing_images(images_dir, coords_dir)

    if mode not in (police_modes+rubber_modes):
        print(f"Неизвестный режим: {mode}")
        print("Доступные режимы: police (find images without coords), rubber (find coords without image), both (Двойной агент: полицейский под прикрытием)")
        sys.exit(1)
