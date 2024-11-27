import cv2;
import os;
import colorsys;


### Constants
wantPrintMessages = False;
wantPrintProgress = True;
wantPrintWarnings = True;

# Расширение для координат
coordsExt = ".txt";
### Конец const


# Функция для загрузки координат из файла
def loadCoords(coordsFile):
    coords = [];
    with open(coordsFile, 'r') as file:
        coords = [tuple(map(int, line.strip().split())) for line in file];
    return coords;

def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v));

# Функция для отображения изображения с разметкой
def showImageWithLandmarks(image, landmarks, putText):
    colors = [0, 17, 22, 27, 31, 36, 42, 48, 60, 69];
    colorCounter = -1;
    colorLen = len(colors);

    lenLandmarks = len(landmarks);
    for i in range(lenLandmarks):
        if (i == colors[colorCounter + 1]):
        	colorCounter += 1;
        curH = colorCounter / colorLen;

        (x, y) = landmarks[i];
        rgb = hsv2rgb(curH, 1, 1);
        dotSize = 4;
        cv2.circle(image, (x, y), dotSize, rgb, -1);
        
        fontScale = 1.5; # good examples: 512*512 — 0.75, 1024*1024 — 1.5
        thickness = 2;
        if (putText == True):
            # cv2.putText(кадр, текст,   координаты, тип шрифта,  масштаб шрифта, цвет [, толщина пера [, тип линии [, центр координат]]])
            cv2.putText(image, str(i+1), (x,y), cv2.FONT_HERSHEY_PLAIN, fontScale, rgb, thickness);

    cv2.imshow("Face Landmarks", image);


# Основная функция для обработки изображений
def processImages(imagePath, coordsPath):
    # Получение списка изображений
    try:
    	imageNames = [f for f in os.listdir(imagePath) if os.path.isfile(os.path.join(imagePath, f))];
    except Exception as e:
    	print(e);
    	print("Use console or terminal from script directory. Press Enter to close.");
    	input();

    currentIndex = 0;
    numImages = len(imageNames);

    while True:
        # Загружаем изображение лица
        imageName = imageNames[currentIndex];
        image = cv2.imread(os.path.join(imagePath, imageName));

        # Получаем имя файла координат на основе имени изображения
        baseName = os.path.splitext(imageName)[0];
        coordsFile = os.path.join(coordsPath, baseName + coordsExt);

        # Проверяем, существует ли файл с координатами
        if (not os.path.exists(coordsFile)) and (wantPrintWarnings):
            print(f"Coordinates file for '{imageName}' not found, skipping...");
            currentIndex = (currentIndex + 1) % numImages;
            continue;

        # Загружаем координаты
        coords = loadCoords(coordsFile);

        # Отображаем изображение с разметкой
        showImageWithLandmarks(image, coords, False);

        # Ожидаем нажатия клавиши
        key = cv2.waitKeyEx(0)
        if (wantPrintMessages):
            print("\tKey: " + str(key));

        # Закрываем окно, если нажата клавиша ESC
        if key == 0x1b:  # 0x1b = 27 — код клавиши ESC
            break;

        # Следующее изображение (D или → = 0x27 = 39)
        if (key == ord('d')) or (key == 0x270000):  
            currentIndex = (currentIndex + 1) % numImages;
        # Предыдущее изображение (A или ← = 0x25 = 37)
        elif (key == ord('a')) or (key == 0x250000):  
            currentIndex = (currentIndex - 1) % numImages;

        # Проверяем, было ли закрыто окно
        if cv2.getWindowProperty("Face Landmarks", cv2.WND_PROP_VISIBLE) < 1:
            break;

    # Закрываем все окна
    cv2.destroyAllWindows();



### Запуск программы
if __name__ == "__main__":
    imagePath = "imagesDone";
    coordsPath = "coordsDone";
    processImages(imagePath, coordsPath);
