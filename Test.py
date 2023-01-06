import numpy as np
from modules.Calc import bbox3d2bev, iou2d
import torch
import math
from shapely.geometry import Polygon
import os
from open3d.cpu.pybind.geometry import PointCloud, OrientedBoundingBox
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries


dataroot = '../mmdetection3d-master/data/kitti'
veloroot = os.path.join(dataroot, 'training/velodyne')
trainInfoPath = os.path.join(dataroot, 'ImageSets/train.txt')
testInfoPath = os.path.join(dataroot, 'ImageSets/val.txt')
labelroot = os.path.join(dataroot, 'training/label_2')
calibroot = os.path.join(dataroot, 'training/calib')

velopath = os.path.join(veloroot, '000295.bin')
labelpath = os.path.join(labelroot, '000295.txt')
calibpath = os.path.join(calibroot, '000295.txt')

velo = np.fromfile(velopath, dtype = 'float32').reshape((-1, 4))
velo[:, 3] = 1
with open(calibpath, 'r') as f:
    l = f.readlines()[5][:-1]
    l = l.split(' ')[1:]
    v2c = np.array(l).astype('float32').reshape((3, 4))
    v2c = np.concatenate([v2c, [[0, 0, 0, 1]]], axis = 0)
    c2v = np.linalg.inv(v2c)
with open(labelpath, 'r') as f:
    l = f.readlines()[2][:-1]
    l = l.split(' ')[8:]
    bbox = np.array(l).astype('float32')


r = bbox[6] - 0.5 * np.pi
bbox[[0, 2]] = bbox[[2, 0]]
xyz = bbox[3:6]
xyz = np.concatenate([xyz, [1]], axis = 0)
xyz = c2v @ xyz
bbox[3:6] = xyz[:3]
rot = torch.Tensor([math.cos(r), -math.sin(r), math.sin(r), math.cos(r)]).reshape((2, 2))
res = torch.Tensor([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])
res = res * bbox[None, [0, 1]]
res = res @ rot
# y1 = torch.empty((4, 1))
# y2 = torch.empty((4, 1))
# y1[...] = 0
# y2[...] = -bbox[1].item()
# y = torch.concat([y1, y2], dim = 0)
z1 = torch.empty((4, 1))
z2 = torch.empty((4, 1))
z1[...] = 0
z2[...] = bbox[2].item()
z = torch.concat([z1, z2], dim = 0)
res = torch.concat([res, res], dim = 0)
res = torch.concat([res, z], dim = 1)
res = res + bbox[None, 3:6]
drawBox = OrientedBoundingBox().create_from_points(Vector3dVector(res))
drawBox.color = (1, 0, 0)

# velo = v2c @ velo.T
# velo = velo.T

pcd = PointCloud(Vector3dVector(velo[:, :3]))
draw_geometries([pcd, drawBox])
