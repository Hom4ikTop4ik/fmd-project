import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Импорт для 3D проекций


cordpath = '../../experiments/condcords00003_3d.txt'
cordfile = open(cordpath, 'r')
ts = torch.zeros(68, 3)
for i, line in enumerate(cordfile):
    ts[i] = torch.tensor(list(map(float, line.split())))
ts[:, 1] = ts[:, 1] * ts[:, 2]
ts[:, 0] = ts[:, 0] * ts[:, 2]

points = ts

# 2. Создание фигуры и 3D проекции:
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# 3. Извлечение координат:
x = points[:, 0].numpy()
y = points[:, 2].numpy()
z = points[:, 1].numpy()

# 4. Построение графика:
ax.scatter(x, y, z, c='r', marker='o')

# Добавление номеров точек
for i in range(len(x)):
    ax.text(x[i], y[i], z[i],  '%s' % (str(i)), size=8, zorder=1,  color='k')

# 5. Настройка осей (необязательно, но рекомендуется):
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Можно установить пределы осей, если нужно:
ax.set_xlim([0.5, 2])  # Пример: пределы для оси X
ax.set_ylim([1.5, 3])  # Пример: пределы для оси Y
ax.set_zlim([0.8, 2.3])  # Пример: пределы для оси Z

# 6. Отображение графика:
plt.title('face points')
plt.show()
