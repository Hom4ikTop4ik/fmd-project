# viewer_loader.py

import os
import cv2
import torch
import numpy as np
import time
from data_process import load, POCHTI, DA, NET, BATCH_SIZE

# масштаб изображения, точек, текста глубины и индексов
SCALE_SCALE = 2
SCALE_LIST = [1.0, 0.5, 0.0, 0.6] # img_, dot_, depth_, index_scale
SCALE = [SCALE_SCALE * i for i in SCALE_LIST]

def interactive_viewer_from_generator(generator, print_i = NET, print_z = NET, delay_secs = 0.01, scale_factors=SCALE):
    cache = []
    coords_cache = []
    index = 0
    
    image_scale, dot_scale, depth_txt_scale, index_txt_scale = scale_factors

    while True:
        if index >= len(cache):
            try:
                img_batch, coords_batch = next(generator)
                for img in img_batch:
                    cache.append(img)
                for coords in coords_batch:
                    coords_cache.append(coords)
            except StopIteration:
                if len(cache) == 0:
                    print("Генератор не дал ни одной картинки.")
                    break
                index = 0  # цикл по модулю
                continue

        image = cache[index]
        coords = coords_cache[index]

        # Преобразуем изображение
        image = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        image = np.ascontiguousarray(image)
        h, w = image.shape[:2]
        
        if image_scale != 1.0:
            image = cv2.resize(image, (int(w * image_scale), int(h * image_scale)))

        # Рисуем точки
        for idx, coord in enumerate(coords):
            x01, y01, z01 = list(map(torch.Tensor.item, coord))
            
            x, y = int(x01 * w * image_scale), int(y01 * h * image_scale)
            z = z01
            
            center = (int(x), int(y))
            radius = int(4 * dot_scale)
            cv2.circle(image, center, radius, (0, 255, 0), -1)

            # текст глубины (право/верх)
            depth_text = f"{z:.2f}"
            depth_pos = (center[0] + int(4 * dot_scale), center[1] + int(6 * dot_scale))
            cv2.putText(image, depth_text, depth_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * depth_txt_scale, (255, 255, 255), 1, cv2.LINE_AA)

            # текст индекса (право/низ)
            index_text = f"{idx}"
            index_pos = (center[0] + int(4 * dot_scale), center[1] - int(3 * dot_scale))
            cv2.putText(image, index_text, index_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * index_txt_scale, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(f"Image", image)


        key = cv2.waitKey(0)

        # 27 — ESC
        if key in [ord('q'), ord('Q'), 27]:
            break
        elif key in [ord('d'), ord('D')]:  # D or Right Arrow
            index += 1
            if index > len(cache):  # Это может произойти при быстром переключении
                index = len(cache)
        elif key in [ord('a'), ord('A')]:  # A or Left Arrow
            index = (index - 1) % len(cache)


        # # Удержание клавиш
        # time.sleep(delay_secs)
        
        # if keyboard.is_pressed('q'):
        #     break
        # elif keyboard.is_pressed('right') or keyboard.is_pressed('d'):
        #     index += 1
        #     if index > len(cache):
        #         index = len(cache)
        # elif keyboard.is_pressed('left') or keyboard.is_pressed('a'):
        #     index = (index - 1) % len(cache)

        # if cv2.waitKey(1) == 27:
        #     break

        


def main():
    dataset_path = "./model/registry/dataset/train"
    
    dataloader, sampler = load(
        bsize=1,  # один кадр за раз
        dataset_path=dataset_path,
        images_dir="images",
        coords_dir="coords",
        augments=NET,
        workers=False
    )

    sampler.set_seed(0)  # зафиксировать порядок

    # for i, (image_batch, coords_batch) in enumerate(dataloader):
    #     image = image_batch[0]
    #     coords = coords_batch[0]

    #     img = draw_coords_on_image(image, coords)

    #     cv2.imshow("Image Viewer", img)
    #     key = cv2.waitKey(0)

    #     if key == 27:  # ESC
    #         break
    #     elif key == ord('s'):
    #         cv2.imwrite(f"frame_{i}.jpg", img)
    #         print(f"Saved frame_{i}.jpg")

    # cv2.destroyAllWindows()

    generator = iter(dataloader)
    interactive_viewer_from_generator(generator, DA, DA)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
