import os
import torch
import numpy as np
import open3d as o3d

def point_display(points):
    '''
    input: (1, num_point, 3) or (num_points, 3)
    '''
    if torch.is_tensor(points):
        points = points.cpu().detach().numpy()
    points = np.squeeze(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def point_save(points, path, file_name, type='ply'):
    '''
    input: (1, num_point, 3) or (num_points, 3)
    type: pcd | ply; pcd is occupies more space
    '''
    if torch.is_tensor(points):
        points = points.cpu().detach().numpy()
    points = np.squeeze(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if not os.path.exists(path):
        print("Creating directory: {}".format(path))
        os.makedirs(path)
    o3d.io.write_point_cloud(os.path.join(path, file_name+'.'+type), pcd, write_ascii=True)


def to_point_cloud(points):
    '''
    input: (1, num_point, 3) or (num_points, 3)
    '''
    if torch.is_tensor(points):
        points = points.cpu().detach().numpy()
    points = np.squeeze(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def to_point_cloud_with_color(points, colors):
    '''
    input: (1, num_point, 3) or (num_points, 3)
    '''
    if torch.is_tensor(points):
        points = points.cpu().detach().numpy()
    if torch.is_tensor(colors):
        colors = colors.cpu().detach().numpy()
    points = np.squeeze(points)
    colors = np.squeeze(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors= o3d.utility.Vector3dVector(colors)
    return pcd


def o3d_point_save(points, path, file_name, type='ply'):
    o3d.io.write_point_cloud(os.path.join(path, file_name+'.'+type), points, write_ascii=True)