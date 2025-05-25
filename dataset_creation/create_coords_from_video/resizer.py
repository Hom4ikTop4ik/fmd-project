import os
import cv2

# Константы размера
TARGET_WIDTH = 512  # X
TARGET_HEIGHT = 512  # Y

# Папка с исходными изображениями
SOURCE_FOLDER = "../../model/registry/dataset/train/images/"

# Папка для сохранения изменённых изображений
OUTPUT_FOLDER = "../../model/registry/dataset/train/images/NORMAL_SIZE"

PRINT_SHORT = "short"
PRINT_LONG = "long"
NO_PRINT = "no_print"
PRINT_TYPE = PRINT_SHORT

def resize_images(source_folder, output_folder, target_width, target_height):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        output_path = os.path.join(output_folder, filename)

        if os.path.isfile(source_path):
            img = cv2.imread(source_path)
            if img is None:
                print(f"Не удалось прочитать файл {filename}, пропускаем")
                continue

            h, w = img.shape[:2]
            if w != target_width or h != target_height:
                if PRINT_TYPE == PRINT_SHORT:
                    print(filename)
                elif PRINT_TYPE == PRINT_LONG:
                    print(f"Изменяем размер {filename} с ({w}, {h}) на ({target_width}, {target_height})")
                resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(output_path, resized)
            
            # else:
                # Если размер совпадает — просто копируем без изменений
                # cv2.imwrite(output_path, img)

if __name__ == "__main__":
    resize_images(SOURCE_FOLDER, OUTPUT_FOLDER, TARGET_WIDTH, TARGET_HEIGHT)
