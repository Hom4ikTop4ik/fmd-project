import os
import numpy as np

def load_points(file_path):
    with open(file_path, 'r') as f:
        points = [list(map(float, line.split())) for line in f]
    return np.array(points)

def save_points(file_path, points):
    with open(file_path, 'w') as f:
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

def calculate_plane_normal(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

def distance_to_plane(point, plane_point, plane_normal):
    vector_to_point = point - plane_point
    return np.dot(vector_to_point, plane_normal)

def project_to_plane(point, plane_point, plane_normal):
    distance = distance_to_plane(point, plane_point, plane_normal)
    return point - distance * plane_normal

def reflect_through_projection(source_point, plane_point, plane_normal):
    projected = project_to_plane(source_point, plane_point, plane_normal)
    to_source = source_point - projected
    reflected = projected - to_source
    return reflected[0], reflected[1]

def adjust_points(input_file, output_file, left_indices, right_indices, plane_indices):
    points = load_points(input_file)
    p1, p2, p3 = points[plane_indices]
    plane_normal = calculate_plane_normal(p1, p2, p3)
    plane_point = p1
    left_points = points[left_indices]
    right_points = points[right_indices]
    left_depth = np.mean(left_points[:, 2])
    right_depth = np.mean(right_points[:, 2])
    left_is_closer = left_depth < right_depth
    corrected_points = points.copy()
    for left_idx, right_idx in zip(left_indices, right_indices):
        if left_is_closer:
            source_idx = left_idx
            target_idx = right_idx
        else:
            source_idx = right_idx
            target_idx = left_idx
        source_point = points[source_idx]
        target_point = points[target_idx]
        new_x, new_y = reflect_through_projection(source_point, plane_point, plane_normal)
        corrected_points[target_idx] = np.array([
            new_x,
            new_y,
            target_point[2]
        ])
    save_points(output_file, corrected_points)