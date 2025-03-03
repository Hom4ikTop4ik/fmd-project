import numpy as np
import pyrender
import torch
import trimesh
import functools

import os
from .flame_pytorch import FLAME, get_config
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

def run_from_this_path(func):
    """
    decorator that changes cwd to the path to this file when executing function
    then returns it back
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        result = func(*args, **kwargs)
        os.chdir(cwd)
        return result
    return wrapper


@run_from_this_path
def convert(
      pose_params: torch.Tensor,
      shape_params: torch.Tensor,
      expression_params: torch.Tensor,
      scl_mv_rot: torch.Tensor):
    '''
    function that converts FLAME parameters to coordinates on camera screen.

    len(pose_params) = 6,
    len(shape_params) = 100,
    len(expression_params) = 50,
    
    scl_mv_rot - tensor for camera position, it
        is torch.Tensor(scale(-1, 1), movex(-1, 1), movey(-1, 1), rotation(-1, 1))
    
    '''

    if hasattr(convert, 'config') == False:
       convert.config = get_config()
       convert.flamelayer = FLAME(convert.config)
       convert.camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    config = convert.config
    flamelayer = convert.flamelayer
    camera = convert.camera

    landmark = None
    if config.optimize_eyeballpose and config.optimize_neckpose:
        neck_pose = torch.zeros(1, 3)
        eye_pose = torch.zeros(1, 6)
        _, landmark = flamelayer(
            shape_params, expression_params, pose_params, neck_pose, eye_pose
        )

    
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [scl_mv_rot[0, 3], 0.0, 1.0 + scl_mv_rot[0, 0], 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ])
    camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
    
    joints = landmark[0].detach().cpu().numpy().squeeze()

    scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2], bg_color=[0.9, 0.9, 0.9])
    scene.add_node(camera_node)

    sm = trimesh.creation.uv_sphere(radius=0.002)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]

    tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    tfs[:, :3, 3] = joints
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)

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

    # Применяем матрицу проекции
    image_vertices_homogeneous = camera_vertices_homogeneous @ projection_matrix.T

    # Делим на w для получения нормализованных координат изображения
    image_vertices = image_vertices_homogeneous[:, :2] / image_vertices_homogeneous[:, 3:]


    movex = scl_mv_rot[0, 1]
    movey = scl_mv_rot[0, 2]
    image_vertices = np.array([
        np.array([pair[0] + movex, pair[1] + movey])
        for pair in image_vertices])
    
    return image_vertices


# image_vertices = convert(torch.zeros(6), torch.zeros(100), torch.zeros(50),
                                #    torch.tensor([0, 0, 0, 0]))

# print(image_vertices)
# draw_and_show_points(image_vertices)
