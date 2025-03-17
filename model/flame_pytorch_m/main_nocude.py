"""
Demo code to load the FLAME Layer and visualise the 3D landmarks on the Face

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.

For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""

import numpy as np
import pyrender
import torch
import trimesh

from flame_pytorch import FLAME, get_config
from PIL import Image, ImageDraw, ImageFont

def draw_and_show_points(points, width=1800, height=1800, font_size=12,
                          color=(255, 255, 255), size=3):
  img = Image.new("RGB", (width, height), (0, 0, 0))
  draw = ImageDraw.Draw(img)
  font = ImageFont.truetype("arial.ttf", size=font_size)
  for i, (x, y) in enumerate(points):
    # scale coordinates from [-1, 1] to [0, width-1] и [0, height-1]
    x_scaled = int((x + 1) / 2 * width)
    y_scaled = int((-y + 1) / 2 * height)
    draw.ellipse((x_scaled - size, y_scaled - size, x_scaled + size, y_scaled + size), fill=color)
    draw.text((x_scaled + size + 2, y_scaled - font_size/2),
               str(i), fill=color, font=font)
  img.show()

config = get_config()
radian = np.pi / 180.0
flamelayer = FLAME(config)

# Creating a batch of mean shapes
shape_params = torch.zeros(8, 100)

# Creating a batch of different global poses
# pose_params_numpy[:, :3] : global rotaation
# pose_params_numpy[:, 3:] : jaw rotaation
pose_params_numpy = np.array(
    [
        [0.0, 0.0, 0.0, 0, 0.0, 0.0],
        [0.0, 0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32)

# Cerating a batch of neutral expressions
expression_params = (torch.rand(8, 50, dtype=torch.float32) - 0.5) * 6
flamelayer

# Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework
vertice, landmark = flamelayer(
    shape_params, expression_params, pose_params
)  # For RingNet project
print(vertice.size(), landmark.size())

if config.optimize_eyeballpose and config.optimize_neckpose:
    neck_pose = torch.zeros(8, 3)
    eye_pose = torch.zeros(8, 6)
    vertice, landmark = flamelayer(
        shape_params, expression_params, pose_params, neck_pose, eye_pose
    )
    print('vertex')
    print('size', landmark.size())
    print(landmark, type(landmark))




# Visualize Landmarks
# This visualises the static landmarks and the pose dependent dynamic landmarks used for RingNet project
faces = flamelayer.faces
for i in range(8):
    
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ])
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    
    # vertices = vertice[i].detach().cpu().numpy().squeeze()
    joints = landmark[i].detach().cpu().numpy().squeeze()
    # vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]

    # tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
    # mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2], bg_color=[0.9, 0.9, 0.9])
    scene.add_node(camera_node)
    # scene.add(mesh)
    sm = trimesh.creation.uv_sphere(radius=0.002)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    # print('sm', sm, '\ntype sm', type(sm))
    tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    tfs[:, :3, 3] = joints
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)

        
    renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

    projection_matrix = camera.get_projection_matrix(width=640, height=480)

    # Преобразование координат вершин

    # Получаем координаты вершин объекта
    object_vertices = joints

    # Преобразуем вершины в однородные координаты (добавляем w=1)
    object_vertices_homogeneous = np.concatenate([object_vertices, np.ones((object_vertices.shape[0], 1))], axis=1)

    # Получаем обратную матрицу преобразования камеры
    camera_world_transform = camera_pose  # это преобразование из мирового пространства в пространство камеры
    world_to_camera = np.linalg.inv(camera_world_transform)

    # Преобразуем вершины из мирового пространства в пространство камеры
    camera_vertices_homogeneous = object_vertices_homogeneous @ world_to_camera.T

    # Делим на w для получения нормализованных координат в пространстве камеры
    camera_vertices = camera_vertices_homogeneous[:, :3] / camera_vertices_homogeneous[:, 3:]

    # Применяем матрицу проекции
    image_vertices_homogeneous = camera_vertices_homogeneous @ projection_matrix.T

    # Делим на w для получения нормализованных координат изображения
    image_vertices = image_vertices_homogeneous[:, :2] / image_vertices_homogeneous[:, 3:]

    print("Координаты вершин на экране:")
    print(image_vertices)

    color, depth = renderer.render(scene)

    image = Image.fromarray(color)
    image.show()
    draw_and_show_points(image_vertices)
    break
