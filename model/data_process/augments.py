import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import random
import time

__all__ = ['make_filter']

frequency = 0.1
show = False

# inpaint дорисовывает углы на основе картинки
# mean заполняет средним цветом картинки (такое себе)
angleFiller = "INPAINT" # or "MEAN"

interpolate_mode = 'bicubic' # or 'linear', 'bilinear', 'nearest'

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

def show_image(img_to_print):
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

def scale_img(img, scale, mode):
    if mode == 'linear':
        scale = torch.nn.functional.interpolate(
            img,
            scale_factor = scale, 
            mode = mode, 
            align_corners = False
        )
    elif mode == 'bilinear':
        scale = torch.nn.functional.interpolate(
            img.unsqueeze(0), # создать псевдо ОСь с размером батча 
            scale_factor = scale, 
            mode = mode, 
            align_corners = False
        ).squeeze(0) # отбросить псевдо ось
    elif mode == 'nearest':
        scale = torch.nn.functional.interpolate(
            img,
            scale_factor = scale, 
            mode = mode
        )
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
        img_resized = cv2.resize(img_cv, (img_cv.shape[1] // SCALE, img_cv.shape[0] // SCALE))  # Уменьшаем в 4 раза

        mask = np.all(img_resized == [0, 0, 0], axis=-1).astype(np.uint8)  # Математическая маска для черных углов

        # Применяем inpainting на уменьшенном изображении
        inpainted_resized_img = cv2.inpaint(img_resized, mask, inpaintRadius=inpaintRadius, flags=cv2.INPAINT_TELEA)

        # Возвращаем изображение в исходный размер
        inpainted_img = cv2.resize(inpainted_resized_img, (img_cv.shape[1], img_cv.shape[0]))
        
        # Наложение на черные углы оригинала
        mask_black_areas = np.all(img_cv == [0, 0, 0], axis=-1)  # Черные области (углы)
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

def augment_image(img, coords, displace: int, rotate=0, noise=0.0, scale=1.0):
    if img.shape[1] != img.shape[2]:
        print("Image is not square!")
        return None, None

    # img = img[[2, 1, 0], :, :]  # Переключаем каналы обратно 
    # но сейчас это делает DataLoader

    # Генерация случайного угла вращения
    angle = (torch.rand(1, device=img.device) * 2 - 1) * rotate  # От -rotate до +rotate
    angle = angle.item()

    if (scale == 1.0):
        pass
    elif (scale < 1.0):
        x, y = img.shape[1], img.shape[2]
        down_scale_img = scale_img(img, scale, mode = interpolate_mode)
        # down_scale_img = F.resize(img, (int(x * scale), int(y * scale)), fill=(0, 0, 0))
        small_x, small_y = down_scale_img.shape[1], down_scale_img.shape[2]
        
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
        up_scale_img = scale_img(img, scale, mode = interpolate_mode)
        # up_scale_img = F.interpolate(img.unsqueeze(0), size=(big_x, big_y), mode=interpolate_mode, align_corners=False).squeeze(0)

        # Обрезка до исходного размера
        cornerx = torch.randint(0, big_x - x, (1,), device=img.device).item()
        cornery = torch.randint(0, big_y - y, (1,), device=img.device).item()
        img_cropped = up_scale_img[:, cornery:cornery + y, cornerx:cornerx + x]
        img = img_cropped
        prevsize = (big_x, big_y)
        coords = crop_coords_big(coords, cornerx, cornerx + x, cornery, cornery + y, prevsize)

    # Заполнение чёрных углов и контуров цветом
    if angleFiller == "MEAN":
        mean_color = img.mean(dim=[1, 2])
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
