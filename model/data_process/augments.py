import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import random
import time
import os
import sys
from datetime import datetime

__all__ = ['make_filter', 'scale_img', 'show_image', 'show_image_coords']

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
colored_shape_use = True
colored_shape_cover_ratio = 0.1
min_colored_shape_slze = 10   # px
max_colored_shape_slze = 300  # px
min_colored_shape_colors = (0, 0, 0)
max_colored_shape_colors = (255, 255, 255)
max_colored_shape_angle = 45  # degrees


# y = A*x + B
LINEAR_ISO_A_FROM = 0.4
LINEAR_ISO_A_TO = 1.6
LINEAR_ISO_B_FROM = -0.05
LINEAR_ISO_B_TO = 0.05

# gamma and offset
LINEAR_ISO_G_FROM = 1
LINEAR_ISO_G_TO = 5
LINEAR_ISO_O_FROM = -0.05
LINEAR_ISO_O_TO = 0.05



def torch_rand_uniform(low, high) -> torch.Tensor:
    """low >= high -> return 0.torch.Tensor()"""
    if (low >= high) :
        return torch.tensor([0])
    return torch.empty(1).uniform_(low, high)

def torch_rand_normal(mean, omega, low=None, high=None) -> torch.Tensor:
    # Генерируем случайное число из стандартного нормального распределения
    x = torch.randn(1) * omega + mean
    
    # Если задан диапазон, ограничиваем значения
    if low is not None and high is not None:
        x.clamp(min=low, max=high)
    
    return x



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

def show_image_numpy(img_to_print):
    """Как и show_image, но принимает ПОЛНОСТЬЮ готовый ndarray
       [torch.Tensor].cpu().numpy().transpose(1, 2, 0)"""
    name = "img_aboba" + str(random.randint(0, 1000))
    cv2.imshow(name, img_to_print)
    cv2.waitKey(0)
    return name

def show_image_coords(img, coords):
    name = "img" + str(random.randint(0, 1000))
    newimg = (img.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

    newimg = np.ascontiguousarray(newimg)
    for coord in coords:
        x, y = coord[0], coord[1]
        newimg = cv2.circle(newimg, (int(x * newimg.shape[1]), int(y * newimg.shape[0])), 2, (255, 255, 255), 2)
    
    cv2.imshow(name, newimg)
    cv2.waitKey(0)



def concat_images_hor(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Склеивает два изображения горизонтально (фото1|фото2).
    
    Аргументы:
    - img1, img2: тензоры изображений с одинаковыми размерами (C, H, W)
    
    Возвращает:
    - тензор изображения (C, H, W1 + W2) [dim = 2]
    """
    assert img1.shape[:2] == img2.shape[:2], "Изображения должны быть одного размера"
    assert img1.dim() == 3, "Ожидается изображение в формате (C, H, W)"
    
    return torch.cat((img1, img2), dim=2)

def concat_images_ver(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Склеивает два изображения вертикально (фото1\nфото2).
    
    Аргументы:
    - img1, img2: тензоры изображений с одинаковыми размерами (C, H, W)
    
    Возвращает:
    - тензор изображения (C, H1 + H2, W)
    """
    assert img1.shape[0] == img2.shape[0] and img1.shape[2] == img2.shape[2], "Изображения должны быть одного размера"
    assert img1.dim() == 3, "Ожидается изображение в формате (C, H, W)"
    
    return torch.cat((img1, img2), dim=1)

def save_tensor_image_cv2(tensor: torch.Tensor, folder: str = ".", 
                          rand_range: tuple = (1000, 9999), ext: str = "png") -> str:
    """
    Сохраняет изображение из тензора в файл с помощью OpenCV (cv2).
    
    Аргументы:
    - tensor: torch.Tensor формата (C, H, W), значения в [0, 1] или [0, 255]
    - folder: путь к папке для сохранения
    - rand_range: диапазон для случайной части имени
    - ext: расширение файла (по умолчанию png)

    Возвращает:
    - Полный путь к сохранённому файлу
    """
    assert tensor.dim() == 3 and tensor.shape[0] in [1, 3], "Ожидается формат (C, H, W)"

    # Переводим в numpy и (H, W, C)
    img = tensor.detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

    # Масштабируем и приводим к uint8
    if img.max() <= 1.0:
        img = img * 255
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Генерируем имя
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_part = random.randint(*rand_range)
    filename = f"{timestamp}_{rand_part}.{ext}"
    filepath = os.path.join(folder, filename)

    # Сохраняем
    cv2.imwrite(filepath, img)

    return filepath




def denormalize_points(landmarks: torch.Tensor, size: int = 512) -> np.ndarray:
    landmarks_2d = landmarks[:, :2]  # Берём только x, y
    return (landmarks_2d * size).int().numpy()

def get_eye_centers(landmarks_px):
    left_eye = landmarks_px[36:42]
    right_eye = landmarks_px[42:48]
    left_center = left_eye.mean(axis=0)
    right_center = right_eye.mean(axis=0)
    return left_center, right_center



def add_gaussian_blur(image_tensor: torch.Tensor, blur_level: int = 5) -> torch.Tensor:
    """
    Добавляет размытие по Гауссу к изображению, имитируя плохую веб-камеру.
    
    image_tensor: [3, H, W], значения от 0 до 1
    blur_level: нечётное число > 1 — размер ядра размытия (чем больше, тем сильнее размытие)
    → возвращает размытое изображение [3, H, W], значения 0..1
    """
    if blur_level < 3 or blur_level % 2 == 0:
        raise ValueError("blur_level должен быть нечётным числом >= 3")
    
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    blurred_np = cv2.GaussianBlur(image_np, (blur_level, blur_level), sigmaX=0)
    blurred_tensor = torch.from_numpy(blurred_np.astype(np.float32) / 255.0).permute(2, 0, 1).clamp(0, 1)
    return blurred_tensor

def simulate_linear_ISO(tensor: torch.Tensor, a: float = 1.0, b: float = 0.0) -> torch.Tensor:
    """y = a*x + b"""
    img = tensor.clone()
    img = a*img + b

    return img.clamp(0, 1)

def simulate_gamma_ISO(tensor: torch.Tensor, gamma: float = 1.0, offset: float = 0.0):
    """y = x^gamma + offset"""
    img = tensor.clone().clamp(min=1e-5)
    result = torch.pow(img, gamma) + offset
    return result.clamp(0, 1)

def simulate_ISO(tensor: torch.Tensor, mode : str = "linear", noise_strength : float = 0.0):
    if mode == "linear":
        a = torch_rand_uniform(LINEAR_ISO_A_FROM, LINEAR_ISO_A_TO)
        b = torch_rand_uniform(LINEAR_ISO_B_FROM, LINEAR_ISO_B_TO)
        img = simulate_linear_ISO(tensor, a, b)
    elif mode == "gamma":
        pre_gamma = torch_rand_uniform(LINEAR_ISO_G_FROM, LINEAR_ISO_G_TO)
        gamma = random.choice([pre_gamma, 1 / pre_gamma])
        offset = torch_rand_uniform(LINEAR_ISO_O_FROM, LINEAR_ISO_O_TO)
        img = simulate_gamma_ISO(tensor, gamma, offset)
    else:
        return tensor

    if noise_strength > 0:
        noise = torch.randn_like(img) * noise_strength
        img = img + noise

    return img


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



def add_colored_shapes(image_tensor: torch.Tensor, cover_ratio: float = 0.07, use_shapes: bool = True) -> torch.Tensor:
    """
    Накладывает случайные повёрнутые цветные прямоугольники, чтобы перекрыть часть изображения.

    image_tensor: [3, H, W], значения 0..1
    cover_ratio: доля перекрытия (например, 0.2 = 20% изображения)
    → возвращает модифицированный тензор [3, H, W], значения 0..1
    """
    if not use_shapes:
        return image_tensor
    
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

def noise_tensor(img, noise_factor, noise_sizes = [1, 2, 4, 8]):
    """
    if noise_tensor empty, no add noise
    Во сколько раз уменьшать картинку, список значений.
    Дальше берётся среднее арифметическое.
    """
    
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

def augment_image(img : torch.Tensor, coords, rotate=0, noise=0.0, scale=1.0, blur_level=5):
    if img.shape[1] != img.shape[2]:
        print("Image is not square!")
        return None, None

    # img = add_glasses_opencv(img, coords, glasses_directory, glasses_probability)
    img = add_colored_shapes(img, colored_shape_cover_ratio, colored_shape_use)
    img = add_gaussian_blur(img, blur_level)
    img = simulate_ISO(img, mode="gamma", noise_strength = 0.05)

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

    return img, coords


def make_filter(*keypoints: list):
    def kpfilter(coordbatch: torch.Tensor):
        return torch.stack([coordbatch[:, k] for k in keypoints]).permute(1, 0, 2)
    return kpfilter