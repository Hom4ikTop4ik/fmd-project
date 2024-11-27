import dlib
import cv2
import os
import shutil


### Constants
wantPrintMessages = False
wantPrintProgress = True
wantPrintWarnings = True

# Поддерживаемые расширения
imageExtensions = ['.bmp', '.jpeg', '.jpg', '.png', '.ppm', '.pgm', '.tif', '.tiff', '.webp']

# Расширение для координат
coordsExt = ".txt"
###



### Init, так сказать
# Загружаем модели для обнаружения лиц...
detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
# ... и определения ключевых точек
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

imagesPath = "images" # Путь до папки с исходными фотографиями
imagesDonePath = "imagesDone" # Готовые фото будут перемещаться сюда для удобства
coordsDonePath = "coordsDone" # Файлы с координатами отдельно, также для удобства

# Убедитесь, что папки существуют
os.makedirs(imagesPath, exist_ok=True)
os.makedirs(imagesDonePath, exist_ok=True)
os.makedirs(coordsDonePath, exist_ok=True)

# Получение списка файлов
imageNames = [f for f in os.listdir(imagesPath) 
               if os.path.isfile(os.path.join(imagesPath, f)) and 
               os.path.splitext(f)[1].lower() in imageExtensions]
### Конец так называемого init



### Обработка каждого изображения
cntOfImageNames = len(imageNames)

for imageIndex  in range(cntOfImageNames):
    if wantPrintProgress:
        print(f"Progress: {100 * imageIndex // cntOfImageNames}% -- {imageIndex}/{cntOfImageNames}")

    curImageName = imageNames[imageIndex]

    ## Пропуск фото
    ## Если фото уже лежит в imagesDone
    ## и координатык нему лежат в coordsDone
    ## я НЕ ХОЧУ перезаписать — skip

    # Получаем имя файла без расширения
    baseName = os.path.splitext(curImageName)[0]
    
    # Только если разметка уже есть
    if os.path.exists(os.path.join(coordsDonePath, baseName + coordsExt)):
        # Проверяем, существует ли файл с таким же именем в imagesDone для всех поддерживаемых расширений
        fileExists = False
        for ext in imageExtensions:
            if os.path.exists(os.path.join(imagesDonePath, baseName + ext)):
                fileExists = True
                break
        
        if fileExists:
            print(f"File '{curImageName}' already processed, skipping...")
            continue
    ## Конец пропуска фото


    ### Загружаем изображение лица
    curImagePath = os.path.join(imagesPath, curImageName)
    image = cv2.imread(curImagePath)

    if (image is None) and (wantPrintWarnings):
        print(f"Can't load '{curImageName}'")
        continue
    ###


    ### Обнаружение лиц c помощью CNN
    # 1 — коэффициент/параметр, который указывает количество увеличений масштаба 
    # изображения перед его обработкой. Этот параметр используется для улучшения 
    # точности обнаружения лиц, особенно когда лица могут быть разного размера.
    dets = detector(image, 1)

    # Проверка, найдены ли лица на изображении
    if (not dets) and (wantPrintWarnings):
        print(f"No face in image '{curImageName}'")
        continue
    ### 



    ### Разметка ключевых точек для найденного лица
    det = dets[0] # Берём только первое найденное лицо

    # Нахождение "прямоугольника" box'а лица
    face = dlib.rectangle(
        int(det.rect.left()), 
        int(det.rect.top()), 
        int(det.rect.right()), 
        int(det.rect.bottom())
    )
    
    # Объект, содержащий координаты ключевых точек лица
    landmarks = predictor(image, face)

    ## Составление пути для файла с координатами
    extToTxt = {
        '.bmp': coordsExt,
        '.jpeg': coordsExt,
        '.jpg': coordsExt,
        '.png': coordsExt,
        '.ppm': coordsExt,
        '.pgm': coordsExt,
        '.tif': coordsExt,
        '.tiff': coordsExt,
        '.webp': coordsExt
    }

    coordsImageName = curImageName
    # Замена всех фото-расширений на coordsExt 
    for ext in extToTxt.keys():
        coordsImageName = coordsImageName.replace(ext, extToTxt[ext])

    coordsPath = os.path.join(coordsDonePath, coordsImageName)
    ## Конец составления пути до файла координат

    # Запись координат в файл
    with open(coordsPath, 'w') as txtfile:
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            txtfile.write(f"{x} {y}\n")
    ### Конец разметки ключевых точек



    ### Перемещение обработанного изображения
    try:
        shutil.move(curImagePath, os.path.join(imagesDonePath, curImageName))
    except Exception as e:
        print(f"Error moving file '{curImageName}': {e}")


    if wantPrintMessages:
        print(f"Processing of {curImageName} is complete, results saved in {coordsPath}.")
### Конец обработки каждого изображения

print(f"Progress: 100% -- {cntOfImageNames}/{cntOfImageNames}")
