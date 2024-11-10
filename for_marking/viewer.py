import cv2
import os


### Constants
wantPrintMessages = False
wantPrintProgress = True
wantPrintWarnings = True

# Расширение для координат
coordsExt = ".txt"
### Конец const


# Функция для загрузки координат из файла
def loadCoords(coordsFile):
    coords = []
    with open(coordsFile, 'r') as file:
        coords = [tuple(map(int, line.strip().split())) for line in file]
    return coords

# Функция для отображения изображения с разметкой
def showImageWithLandmarks(image, landmarks):
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    cv2.imshow("Face Landmarks", image)

# Основная функция для обработки изображений
def processImages(imagePath, coordsPath):
    # Получение списка изображений
    imageNames = [f for f in os.listdir(imagePath) if os.path.isfile(os.path.join(imagePath, f))]

    currentIndex = 0
    numImages = len(imageNames)

    while True:
        # Загружаем изображение лица
        imageName = imageNames[currentIndex]
        image = cv2.imread(os.path.join(imagePath, imageName))

        # Получаем имя файла координат на основе имени изображения
        baseName = os.path.splitext(imageName)[0]
        coordsFile = os.path.join(coordsPath, baseName + coordsExt)

        # Проверяем, существует ли файл с координатами
        if (not os.path.exists(coordsFile)) and (wantPrintWarnings):
            print(f"Coordinates file for '{imageName}' not found, skipping...")
            currentIndex = (currentIndex + 1) % numImages
            continue

        # Загружаем координаты
        coords = loadCoords(coordsFile)

        # Отображаем изображение с разметкой
        showImageWithLandmarks(image, coords)

        # Ожидаем нажатия клавиши
        key = cv2.waitKeyEx(0)
        if (wantPrintMessages):
            print("\tKey: " + str(key))

        # Закрываем окно, если нажата клавиша ESC
        if key == 0x1b:  # 0x1b = 27 — код клавиши ESC
            break

        # Следующее изображение (D или → = 0x27 = 39)
        if (key == ord('d')) or (key == 0x270000):  
            currentIndex = (currentIndex + 1) % numImages
        # Предыдущее изображение (A или ← = 0x25 = 37)
        elif (key == ord('a')) or (key == 0x250000):  
            currentIndex = (currentIndex - 1) % numImages

        # Проверяем, было ли закрыто окно
        if cv2.getWindowProperty("Face Landmarks", cv2.WND_PROP_VISIBLE) < 1:
            break

    # Закрываем все окна
    cv2.destroyAllWindows()



### Запуск программы
if __name__ == "__main__":
    imagePath = "imagesDone"
    coordsPath = "coordsDone"
    processImages(imagePath, coordsPath)
