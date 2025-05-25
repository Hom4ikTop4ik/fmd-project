import cv2
import numpy as np



def draw_3d_points(image_path, coord_path, SCALE=[1.0, 1.0, 1.0, 1.0]):
    """
    SCALE = [image_scale, point_scale, depth_text_scale, index_text_scale]
    """

    # === Распаковка масштабов ===
    image_scale     = SCALE[0]  # масштаб размера изображения
    point_scale     = SCALE[1]  # радиус точки
    depth_txt_scale = SCALE[2]  # размер текста глубины
    index_txt_scale = SCALE[3]  # размер текста индекса

    # === Загрузка изображения ===
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить {image_path}")
        return

    height, width = image.shape[:2]

    # === Загрузка координат ===
    coords = []
    with open(coord_path, "r") as f:
        for line in f:
            x, y, z = map(float, line.strip().split())
            coords.append((x * width, y * height, z))

    # === Увеличение изображения ===
    if image_scale != 1.0:
        image = cv2.resize(image, (int(width * image_scale), int(height * image_scale)))
        # Обновить размеры
        height, width = image.shape[:2]
        coords = [(x * image_scale, y * image_scale, z) for x, y, z in coords]

    # === Отображение точек ===
    for idx, (x, y, z) in enumerate(coords):
        center = (int(x), int(y))
        radius = int(4 * point_scale)
        cv2.circle(image, center, radius, (0, 255, 0), -1)

        # текст глубины (право/верх)
        depth_text = f"{z:.2f}"
        depth_pos = (center[0] + int(5 * point_scale), center[1] - int(8 * point_scale))
        cv2.putText(image, depth_text, depth_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * depth_txt_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # текст индекса (право/низ)
        index_text = f"{idx}"
        index_pos = (center[0] + int(5 * point_scale), center[1] + int(15 * point_scale))
        cv2.putText(image, index_text, index_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4 * index_txt_scale, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Shower", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# === Пример использования ===
if __name__ == "__main__":
    num = 27269
    image_path = f"output_1_1/images/dataimg{num}.jpg"
    coord_path = f"output_1_1/coords/condcords{num}_3d.txt"
    image_path = f"I:/fmd-project/model/registry/dataset/train/images_Liliput/dataimg{num}.jpg"
    coord_path = f"I:/fmd-project/model/registry/dataset/train/coords_Liliput/condcords{num}_3d.txt"
    draw_3d_points(image_path, coord_path, [2.0, 1, 0, 0])