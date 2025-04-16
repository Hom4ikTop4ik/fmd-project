import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import random
import time
import os
import sys

__all__ = ['make_filter', 'scale_img', 'show_image', 'show_image2']

# inpaint дорисовывает углы на основе картинки
# mean заполняет средним цветом картинки (такое себе)
# (mean оставляет странный средний чёрный квадрат) 
angleFiller = "INPAINT" # "INPAINT" or "MEAN"

interpolate_mode = 'bilinear' # 'nearest','bicubic' or 'bilinear'. bi___ because picture is 2D :(
# nearest - the fastest and the shakal'est

glasses_list = []
glasses_probability = 0.0
glasses_directory = "I:/fmd-project/model/glasses"

    # --- Локальные параметры ---
colored_shape_cover_ratio = 0.1
min_colored_shape_slze = 10   # px
max_colored_shape_slze = 300  # px
min_colored_shape_colors = (0, 0, 0)
max_colored_shape_colors = (255, 255, 255)
max_colored_shape_angle = 45  # degrees


def rotate_coords(coords, angle_degrees):
    angle_radians = -torch.deg2rad(torch.tensor(angle_degrees, dtype=torch.float32))
    cos_t = torch.cos(angle_radians)
    sin_t = torch.sin(angle_radians)
    
    rot_mat = torch.tensor([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])

    for i in range(coords.shape[0]):
        x, y = coords[i][0], coords[i][1]
        x -= 0.5
        y -= 0.5
        vecp = torch.tensor([x, y])
        vecp = torch.matmul(rot_mat, vecp)
        coords[i][0] = vecp[0] + 0.5
        coords[i][1] = vecp[1] + 0.5
    return coords

def crop_coords_big(coords, left, right, top, bottom, prevsize):
    for i in range(coords.shape[0]):
        x = coords[i][0] * prevsize[0]
        y = coords[i][1] * prevsize[1]
        x -= left
        y -= top
        coords[i][0] = x / float(right - left)
        coords[i][1] = y / float(bottom - top)
    return coords

def crop_coords_small(coords, left, right, top, bottom, prevsize):
    for i in range(coords.shape[0]):
        x = (coords[i][0] * (right - left) + left) / prevsize[0]
        y = (coords[i][1] * (bottom - top) + top) / prevsize[1]
        coords[i][0] = x
        coords[i][1] = y
    return coords

def show_image(img_to_print: torch.Tensor):
    name = "img" + str(random.randint(0, 1000))
    cv2.imshow(name, img_to_print.cpu().numpy().transpose(1, 2, 0))
    cv2.waitKey(0)
    # Для продолжения жмякнуть клавишу в любом активном окне с фото, 
    #   чтобы продолжить работу НЕ закрывая окно 
    # (возможны временные зависания)
    return name

def show_image2(img, coords):
    name = "img" + str(random.randint(0, 1000))
    newimg = (img.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

    newimg = np.ascontiguousarray(newimg)
    for coord in coords:
        x, y = coord[0], coord[1]
        newimg = cv2.circle(newimg, (int(x * newimg.shape[1]), int(y * newimg.shape[0])), 2, (255, 255, 255), 2)
    
    cv2.imshow(name, newimg)
    cv2.waitKey(0)



def denormalize_points(landmarks: torch.Tensor, size: int = 512) -> np.ndarray:
    landmarks_2d = landmarks[:, :2]  # Берём только x, y
    return (landmarks_2d * size).int().numpy()

def get_eye_centers(landmarks_px):
    left_eye = landmarks_px[36:42]
    right_eye = landmarks_px[42:48]
    left_center = left_eye.mean(axis=0)
    right_center = right_eye.mean(axis=0)
    return left_center, right_center

def load_random_glasses(glasses_dir):
    global glasses_list

    # оптимизируем: очков не так много — сохраним в список, чтоб постоянно не подгружать
    if glasses_list == []:
        files = [f for f in os.listdir(glasses_dir) if f.lower().endswith(".png")]
        if not files:
            raise FileNotFoundError("No glasses images *.png in directory!")
        for file in files:
            file_path = os.path.join(glasses_dir, file)
            glasses_list.append(cv2.imread(file_path, cv2.IMREAD_UNCHANGED))

    return random.choice(glasses_list)

def transform_glasses_cv2(glasses, angle, scale):
    h, w = glasses.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    transformed = cv2.warpAffine(glasses, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return transformed

# return image + glasses 
def overlay_glasses(image, glasses, x, y):
    h, w = glasses.shape[:2]
    ih, iw = image.shape[:2]

    # Координаты вставки на изображение
    x1 = max(x - w // 2, 0)
    y1 = max(y - h // 2, 0)
    x2 = min(x1 + w, iw)
    y2 = min(y1 + h, ih)

    # Соответствующий участок очков
    gx1 = max(0, - (x - w // 2))
    gy1 = max(0, - (y - h // 2))
    gx2 = gx1 + (x2 - x1)
    gy2 = gy1 + (y2 - y1)

    roi = image[y1:y2, x1:x2]
    glasses_crop = glasses[gy1:gy2, gx1:gx2]

    # Разделяем BGR и альфа-канал
    alpha = glasses_crop[:, :, 3:4] / 255.0
    bgr = glasses_crop[:, :, :3].astype(np.float32)

    # Наложение по альфа-каналу
    blended = (1 - alpha) * roi.astype(np.float32) + alpha * bgr
    image[y1:y2, x1:x2] = blended.astype(np.uint8)
    return image

def add_glasses_opencv(image_tensor, landmarks, glasses_dir, probability=0.0):
    if random.random() > probability:
        return image_tensor # пропускаем

    image = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # [H, W, C] в uint8
    landmarks_px = denormalize_points(landmarks)

    left_center, right_center = get_eye_centers(landmarks_px)
    eye_center = ((left_center + right_center) / 2).astype(int)
    eye_dist = np.linalg.norm(right_center - left_center)

    # Аугментации
    scale_jitter = random.uniform(0.9, 1.1)
    rotation_jitter = random.uniform(-1, 1)
    shift_x = random.randint(-10, 10)
    shift_y = random.randint(-10, 10)

    glasses = load_random_glasses(glasses_dir)
    scale = (eye_dist * 2.0 / glasses.shape[1]) * scale_jitter
    glasses = transform_glasses_cv2(glasses, angle=rotation_jitter, scale=scale)

    # Вставка
    image = overlay_glasses(image, glasses, eye_center[0] + shift_x, eye_center[1] + shift_y)

    # Обратно в тензор
    image_tensor_out = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).clamp(0, 1)
    return image_tensor_out



def add_colored_shapes(image_tensor: torch.Tensor, cover_ratio: float = 0.07) -> torch.Tensor:
    """
    Накладывает случайные повёрнутые цветные прямоугольники, чтобы перекрыть часть изображения.

    image_tensor: [3, H, W], значения 0..1
    cover_ratio: доля перекрытия (например, 0.2 = 20% изображения)
    → возвращает модифицированный тензор [3, H, W], значения 0..1
    """
    image = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    h, w = image.shape[:2]
    total_area = h * w
    target_area = total_area * cover_ratio
    covered_area = 0


    while covered_area < target_area:
        # Случайный размер прямоугольника
        rect_w = random.randint(min_colored_shape_slze, max_colored_shape_slze)
        rect_h = random.randint(min_colored_shape_slze, max_colored_shape_slze)
        rect_area = rect_w * rect_h

        # пробуем прямоугольник поменбше
        if (covered_area + rect_area > target_area):
            rect_w //= 2
            rect_h //= 2
        rect_area = rect_w * rect_h
        if (covered_area + rect_area > target_area):
            break

        # Случайный цвет
        color = tuple([
            random.randint(min_colored_shape_colors[i], max_colored_shape_colors[i])
            for i in range(3)
        ])

        # Случайная позиция
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        angle = random.uniform(-max_colored_shape_angle, max_colored_shape_angle)

        # Создаём белую маску с одним прямоугольником
        box = cv2.boxPoints(((cx, cy), (rect_w, rect_h), angle))
        box = np.intp(box)
        cv2.drawContours(image, [box], 0, color, thickness=cv2.FILLED)

        covered_area += rect_area

    image_tensor_out = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).clamp(0, 1)
    return image_tensor_out




def scale_img(img, scale, mode = 'bilinear'):
    if mode == 'bilinear':
        scale = torch.nn.functional.interpolate(
            img.unsqueeze(0), # создать псевдо ОСь с размером батча 
            scale_factor = scale, 
            mode = mode, 
            align_corners = False
        ).squeeze(0) # отбросить псевдо ось
    elif mode == 'nearest':
        scale = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            scale_factor = scale, 
            mode = mode
        ).squeeze(0)
    elif mode == 'bicubic':
        scale = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            scale_factor = scale, 
            mode = mode, 
            align_corners = False
        ).squeeze(0)
    
    return scale

def noise_tensor(img, noise_factor, verbose = False):
    if verbose:
        print("augments.py/noise_tensor/device:" + str(img.device))

    # Во сколько раз уменьшать картинку, список значений.
    # Дальше берётся среднее арифметическое.
    noise_sizes = [1, 2, 4, 8] # if empty, no add noise
    
    img_copy = img
    
    for noise_size in noise_sizes:
        # # Уменьшаем изображение
        downscaled = scale_img(img_copy, 1.0 / noise_size, mode = interpolate_mode)

        # Добавляем шум в уменьшенное изображение
        noise_small = torch.randn_like(downscaled, device=img.device) * noise_factor
        downscaled = downscaled + noise_small
        # downscaled = torch.clamp(downscaled, 0, 1)
    
        # # Увеличиваем обратно
        upscaled = scale_img(downscaled, noise_size, mode = interpolate_mode)
        
        img = img + upscaled

    img = torch.clamp(img / float(len(noise_sizes) + 1), 0, 1)

    return img

def in_paint(img):
    SCALE = 32
    inpaintRadius = 1
    # Преобразуем тензор в изображение для использования с OpenCV
    img_cv = img.permute(1, 2, 0).cpu().numpy()  # Переупорядочиваем каналы (C x H x W -> H x W x C)
    img_cv = np.clip(img_cv*255, 0, 255).astype(np.uint8)  # Преобразуем в формат uint8

    if SCALE != 1:
        img_resized = cv2.resize(img_cv, (img_cv.shape[1] // SCALE, img_cv.shape[0] // SCALE))

        mask = np.all(img_resized == [0, 0, 0], axis=-1).astype(np.uint8)  # Математическая маска для черных углов

        inpainted_small_img = cv2.inpaint(img_resized, mask, inpaintRadius=inpaintRadius, flags=cv2.INPAINT_TELEA)

        inpainted_img = cv2.resize(inpainted_small_img, (img_cv.shape[1], img_cv.shape[0]))
        
        # Наложение на черные области оригинала
        mask_black_areas = np.all(img_cv == [0, 0, 0], axis=-1)  # Черные области
        img_cv[mask_black_areas] = inpainted_img[mask_black_areas]

        # Преобразуем обратно в формат PyTorch
        img = torch.from_numpy(img_cv).permute(2, 0, 1).float().to(img.device) / 255.0
    else:
        # Создаем маску для пустых (черных) областей
        mask = np.all(img_cv == [0, 0, 0], axis=-1).astype(np.uint8) * 255

        # Применяем inpainting
        inpainted_img = cv2.inpaint(img_cv, mask, inpaintRadius=inpaintRadius, flags=cv2.INPAINT_TELEA)
        inpainted_img = np.clip(inpainted_img, 0, 255).astype(np.uint8)  # Убедимся, что значения в допустимом диапазоне
        
        # Преобразуем результат обратно в тензор
        img = torch.from_numpy(inpainted_img).permute(2, 0, 1).float().to(img.device) / 255.0  # H x W x C -> C x H x W и нормализация
    return img

def augment_image(img, coords, rotate=0, noise=0.0, scale=1.0):
    if img.shape[1] != img.shape[2]:
        print("Image is not square!")
        return None, None

    # img = img[[2, 1, 0], :, :]  # Переключаем каналы обратно 
    # но сейчас это делает DataLoader

    img = add_glasses_opencv(img, coords, glasses_directory, glasses_probability)
    img = add_colored_shapes(img, colored_shape_cover_ratio)

    # Генерация случайного угла вращения
    angle = (torch.rand(1, device=img.device) * 2 - 1) * rotate  # От -rotate до +rotate
    angle = angle.item()

    if angleFiller == "MEAN":
        mean_color = img.mean(dim=[1, 2])

    if (scale == 1.0):
        pass
    elif (scale < 1.0):
        x, y = img.shape[1], img.shape[2]
        small_x = int(x * scale)
        small_y = int(y * scale)

        if (x > small_x):
            down_scale_img = scale_img(img, scale, mode = interpolate_mode)
            # down_scale_img = F.resize(img, (int(x * scale), int(y * scale)), fill=(0, 0, 0))
            
            cornerx = torch.randint(0, x - small_x, (1,), device=img.device).item()
            cornery = torch.randint(0, y - small_y, (1,), device=img.device).item()

            background = torch.zeros_like(img)
            background[:, cornery:cornery + small_y, cornerx:cornerx + small_x] = down_scale_img
            img = background
            prevsize = (x, y)
            coords = crop_coords_small(coords, cornerx, cornerx + small_x, cornery, cornery + small_y, prevsize)
    elif (scale > 1.0):
        x, y = img.shape[1], img.shape[2]

        big_x = int(x * scale)
        big_y = int(y * scale)
        if (big_x > x):
            up_scale_img = scale_img(img, scale, mode = interpolate_mode)

            # Обрезка до исходного размера
            cornerx = torch.randint(0, big_x - x, (1,), device=img.device).item()
            cornery = torch.randint(0, big_y - y, (1,), device=img.device).item()
            img_cropped = up_scale_img[:, cornery:cornery + y, cornerx:cornerx + x]
            img = img_cropped
            prevsize = (big_x, big_y)
            coords = crop_coords_big(coords, cornerx, cornerx + x, cornery, cornery + y, prevsize)

    # Заполнение чёрных углов и контуров цветом
    if angleFiller == "MEAN":
        img = F.rotate(img, angle, fill=tuple(mean_color))
        
    elif angleFiller == "INPAINT":
    # Заменяем средний цвет на черный (для удобства)
        black_color = (0, 0, 0)  # Черный цвет для заполнения
        img = F.rotate(img, angle, fill=black_color)
        img = in_paint(img) # instead of mean color
    
    # Вращение координат
    coords = rotate_coords(coords, angle)

    # Добавление шума
    if noise > 0:
        img = noise_tensor(img, noise)

    # only if Ilya wants, but it still works without 432px
    # img = scale_img(img, 432/512, interpolate_mode)

    return img, coords


def make_filter(*keypoints: list):
    def kpfilter(coordbatch: torch.Tensor):
        return torch.stack([coordbatch[:, k] for k in keypoints]).permute(1, 0, 2)
    return kpfilter