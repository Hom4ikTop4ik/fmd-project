import dlib
import cv2

# Загружаем модель обнаружения лиц (CNN)
detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
# Загружаем модель определения ключевых точек
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Загружаем изображение лица
image = cv2.imread("generated_face.jpg")

# Обнаружение лиц с помощью CNN
dets = detector(image, 1) 

# Разметка ключевых точек для каждого найденного лица
for det in dets:
    # Преобразование прямоугольника CNN-модели в прямоугольник Dlib
    face = dlib.rectangle(int(det.rect.left()), int(det.rect.top()), 
                           int(det.rect.right()), int(det.rect.bottom()))
    
    landmarks = predictor(image, face)

    # 68 точек лица в формате (x, y)
    for i in range(68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        # Рисуем точку на изображении
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# Показываем изображение с разметкой
cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
