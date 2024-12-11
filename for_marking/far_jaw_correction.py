import os
import numpy as np

# Загрузка данных из файла
def load_points(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")
    with open(file_path, 'r') as f:
        points = [list(map(float, line.split())) for line in f]
    return np.array(points)

# Сохранение данных в файл
def save_points(file_path, points):
    with open(file_path, 'w') as f:
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

# Вычисление нормали к плоскости
def calculate_plane_normal(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

# Вычисление расстояния точки до плоскости
def distance_to_plane(point, plane_point, plane_normal):
    vector_to_point = point - plane_point
    return np.dot(vector_to_point, plane_normal)

# Отражение точки относительно плоскости с сохранением значения глубины
def reflect_point(point, plane_point, plane_normal):
    distance = distance_to_plane(point, plane_point, plane_normal)
    reflected = point - 2 * distance * plane_normal
    # Возвращаем точку с оригинальным значением глубины
    return np.array([reflected[0], reflected[1], point[2]])

# Главная функция корректировки точек
def adjust_points(input_file, output_file, left_indices, right_indices, plane_indices):
    points = load_points(input_file)

    # Выделяем точки для построения плоскости
    p1, p2, p3 = points[plane_indices]
    plane_normal = calculate_plane_normal(p1, p2, p3)
    plane_point = p1

    # Определяем, какая половина ближе к камере
    left_points = points[left_indices]
    right_points = points[right_indices]
    left_depth = np.mean(left_points[:, 2])
    right_depth = np.mean(right_points[:, 2])
    left_is_closer = left_depth < right_depth

    # Корректируем координаты
    corrected_points = points.copy()
    for left_idx, right_idx in zip(left_indices, right_indices):
        if left_is_closer:
            source_idx = left_idx
            target_idx = right_idx
        else:
            source_idx = right_idx
            target_idx = left_idx

        # Исходные точки
        source_point = points[source_idx]
        target_point = points[target_idx]

        # Вычисляем расстояние до плоскости симметрии
        distance = distance_to_plane(source_point, plane_point, plane_normal)

        # Отражаем точку относительно плоскости симметрии
        reflected = reflect_point(source_point, plane_point, plane_normal)
        
        # Обновляем координаты
        corrected_points[target_idx] = np.array([
            reflected[0],  # новый x
            reflected[1],  # новый y
            target_point[2]  # оригинальный z (глубина)
        ])

    # Сохраняем результат
    save_points(output_file, corrected_points)


# Пример вызова функции
if __name__ == "__main__":
    input_file = "face_points.txt"
    output_file = "corrected_face_points.txt"

    # Соответствие точек левой и правой половин лица
    # Формат: (левая точка, правая точка)
    point_pairs = [
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
        # (21, 22), (20, 23), (19, 24), (18, 25), (17, 26),
        # (39, 42), (38, 43), (37, 44), (36, 45), (41, 47), (40, 47),
        # (31, 35), (32, 34),
        # (50, 52), (49, 53), (48, 54), (50, 64),
        # (59, 55), (58, 56)
    ]

    # Точки, лежащие посередине лица (не имеют пары)
    middle_points = [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]

    # Разделяем пары на левые и правые индексы
    left_indices = [pair[0] for pair in point_pairs]
    right_indices = [pair[1] for pair in point_pairs]

    # Индексы для построения плоскости (точки 27, 30 и 8)
    plane_indices = [27, 30, 8]

    try:
        adjust_points(input_file, output_file, left_indices, right_indices, plane_indices)
        print(f"Координаты успешно скорректированы и сохранены в {output_file}")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
