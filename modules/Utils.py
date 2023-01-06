import pickle as pkl
import cvbase as cvb
from pycocotools import mask as maskUtil
import numpy as np
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.geometry import PointCloud, OrientedBoundingBox
from open3d.visualization import draw_geometries
import cv2
from mmdet.core import bbox_overlaps
from mmdet3d.core import bbox_overlaps_3d, CameraInstance3DBoxes, Box3DMode, LiDARInstance3DBoxes
import torch

def readPkl(path: str):
    """
    Read .pkl file from path
    @param path: Path of file
    @return: Content
    """
    with open(path, 'rb') as f:
        res = pkl.load(f)
    return res

def writePkl(path: str, data):
    with open(path, 'wb') as f:
        pkl.dump(data, f)

def readSegJson(path: str):
    """
    Read segmentation label file in .json format from KINS dataset
    @param path: Path of file
    @return: Two dicts, detailed information can be found in vis_json_2020.py, a demo from original author
    """
    json = cvb.load(path)
    imgs = json['images']
    anns = json["annotations"]

    imgs_dict = {}
    anns_dict = {}
    for ann in anns:
        image_id = ann["image_id"]
        if not image_id in anns_dict:
            anns_dict[image_id] = []
            anns_dict[image_id].append(ann)
        else:
            anns_dict[image_id].append(ann)

    for img in imgs:
        image_id = img['id']
        imgs_dict[image_id] = img['file_name']

    return imgs_dict, anns_dict

def polys_to_mask(polygons, height, width):
    """
    Usage can be found in demo
    @param polygons:
    @param height:
    @param width:
    @return:
    """
    rles = maskUtil.frPyObjects(polygons, height, width)
    rle = maskUtil.merge(rles)
    mask = maskUtil.decode(rle)
    return mask

def showNumpyPointCloud(velo: np.ndarray, box3ds = None, cam = True) -> None:
    """
    Visualize a point cloud in numpy format by open3d, draw boxes for option
    @param box3ds: (N,7+C) array-like object, which is the 3d bboxes to show. None if no box
    @param velo: Point cloud in shape of (N, M), where M >= 3 and (N, 3) part of the points is their coordinates
    """
    assert velo.ndim == 2, \
        'The point cloud in numpy required shape of (N, M), where M >= 3 and the coordinate is the first 3 columns'
    pcd = PointCloud(Vector3dVector(velo[:, :3]))
    toshow = [pcd]
    if box3ds is not None:
        if cam:
            box3ds = Box3DMode.convert(box3ds, Box3DMode.CAM, Box3DMode.LIDAR)
        boxins = LiDARInstance3DBoxes(box3ds)
        corners = boxins.corners
        for c in corners:
            showbox = OrientedBoundingBox().create_from_points(Vector3dVector(c))
            showbox.color = (1, 0, 0)
            toshow.append(showbox)
    draw_geometries(toshow)

def veloToImage(velo: np.ndarray, calib: dict, needTranspose = False):
    """
    Project point cloud in LiDAR coordinates to 2d camera image
    @param velo: Point cloud in (4, N), or (N, 4)
    @param calib: Calibration, the matrix in it should be preprocessed to (4, 4)
    @param needTranspose: Set to True if point cloud is in shape of (N, 4)
    @return: Image in numpy format, the points would be truncated to integer, using reflectence as its color.
            Secondly return processed point cloud, in which the points out of image bound are discarded
    """
    assert velo.ndim == 2, 'Point cloud should be in (4, N) or (N, 4)'
    velo_ = velo.T.copy()
    if needTranspose:
        velo = velo.T
        velo_ = velo_.T
    velo = calib['R0_rect'] @ calib['Tr_velo_to_cam'] @ velo
    overzero = velo[2] >= 0
    velo_ = velo_[overzero]
    velo = velo[:, overzero] # 去掉深度小于0的点（在摄像机后面）
    velo = calib['P2'] @ velo
    velo[:2] = velo[:2] / velo[2]
    velo = velo[[0, 1, 3]].T
    filter = (velo[:, 0] >= 0) & (velo[:, 0] < 1242) & (velo[:, 1] >= 0) & (velo[:, 1] < 375)
    velo = velo[filter]
    board = np.ones((375, 1242), dtype = np.float32)
    for p in velo:
        # board先宽后长 此处x，y分别为长，宽坐标，所以反过来
        x = int(p[0])
        y = int(p[1])
        board[y, x] = p[2]
    return board, velo_[filter]

def veloToCam3d(velo: np.ndarray, calib: dict, needTranspose = False, discardUnderZero = True) -> np.ndarray:
    """
    Transfer point cloud from LiDAR coordinates to camera coordinates
    @param velo: Point cloud in (4, N) or (N, 4)
    @param calib: Calibration, the matrix in it should be preprocessed to (4, 4)
    @param needTranspose: Set to True if the point cloud is in shape of (N, 4)
    @param discardUnderZero: Whether to discard the points with z coordinate under zero, i.e., behind the camera
    @return: Point cloud in camera coordinates
    """
    assert velo.ndim == 2, 'Point cloud should be in (4, N) or (N, 4)'
    if needTranspose:
        velo = velo.T
    velo = calib['R0_rect'] @ calib['Tr_velo_to_cam'] @ velo
    if discardUnderZero:
        velo = velo[:, velo[2] >= 0]  # 去掉深度小于0的点（在摄像机后面）
    velo = calib['P2'] @ velo
    return velo

from typing import Iterable
def expandCalib(calib: dict, keys: Iterable[str] = None) -> None:
    """
    Expand the 3x4 or 3x3 matrices in calibration to 4x4, by filling 0 except [3, 3] being 1, in place
    @param calib: Calibration
    @param keys: If specified, calib[key] would be expanded only if key is in keys. Default None, meaning every key would be expanded
    """
    if keys is None:
        keys = calib.keys()
    for key in keys:
        mat = np.zeros((4, 4), dtype = np.float32)
        old = calib[key]
        h, w = old.shape
        mat[:h, :w] = old
        mat[3, 3] = 1
        calib[key] = mat

def paste(velo, img, velogt, imggt, mask, bbox2d):
    velo = np.concatenate([velo, velogt], axis = 0)
    alignedBox = np.round(bbox2d).astype('int32').tolist()
    imgroi = img[alignedBox[1]:alignedBox[3], alignedBox[0]:alignedBox[2]]
    gray = cv2.cvtColor(imggt, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # print(mask)
    maskInv = cv2.bitwise_not(mask)
    imgroi = cv2.bitwise_and(imgroi, imgroi, mask = maskInv)
    img[alignedBox[1]:alignedBox[3], alignedBox[0]:alignedBox[2]] = cv2.add(imgroi, imggt)
    u = max(0, alignedBox[1] - 20)
    d = min(img.shape[0], alignedBox[3] + 20)
    l = max(0, alignedBox[0] - 20)
    r = min(img.shape[1], alignedBox[2] + 20)
    imgroi = img[u:d, l:r]
    imgroi = cv2.GaussianBlur(imgroi, (3, 3), 0)
    img[u:d, l:r] = imgroi
    return velo, img

import random as rd
thrs = (0, 0.3, 0.5, 0.7)
cls = {'Car', 'Cyclist', 'Pedestrian'}
def enhancePaste(scenevelo, sceneimg, scenelabels, gts):
    n = len(gts)
    gcls = []
    bbox2ds = []
    bbox3ds = []
    for g in gts:
        gcls.append(g['class'])
        bbox2ds.append(g['bbox2d'])
        bbox3ds.append(g['bbox3d'])
    gcls = np.array(gcls)
    bbox2ds = torch.Tensor(bbox2ds)
    bbox3ds = torch.Tensor(bbox3ds)
    chosen = []
    cnt = dict(Car = 0, Cyclist = 0, Pedestrian = 0)
    lim = dict(Car = 12, Cyclist = 6, Pedestrian = 6)
    clsfil = np.array([True] * n)
    s = 0
    for c in scenelabels['class']:
        if c not in cls:
            continue
        cnt[c] += 1
        s += 1
    while s < 24:
        thr = thrs[rd.randint(0, 3)]
        # thr = 1
        iou3ds = torch.max(bbox_overlaps_3d(
            torch.Tensor(scenelabels['bbox3d']),
            bbox3ds
        ), dim = 0)[0]
        iou2ds = torch.max(bbox_overlaps(
            torch.Tensor(scenelabels['bbox2d']),
            bbox2ds,
            mode = 'iof'
        ), dim = 0)[0]
        iou2ds2 = torch.max(bbox_overlaps(
            bbox2ds,
            torch.Tensor(scenelabels['bbox2d']),
            mode = 'iof'
        ), dim = 1)[0]
        # print(iou3ds)
        # print(iou2ds)
        # print(torch.where(iou3ds < 0.05), torch.where(iou2ds < thr))
        poses = torch.where((iou3ds < 0.02) & (iou2ds <= thr) & (iou2ds2 <= thr) & clsfil)[0]
        if len(poses) == 0:
            continue
        # print('augmented')
        # print(poses)
        r = rd.randint(0, len(poses) - 1)
        gt = gts[int(poses[r])]
        # if cnt[gt['class']] == lim[gt['class']]:
        #     continue
        chosen.append(gt)
        cnt[gt['class']] += 1
        if cnt[gt['class']] == lim[gt['class']]:
            clsfil = clsfil & (gcls != gt['class'])
        scenelabels['class'] = np.concatenate([scenelabels['class'], np.array([gt['class']])], axis = 0)
        scenelabels['bbox2d'] = np.concatenate([scenelabels['bbox2d'], gt['bbox2d'].reshape((1, -1))], axis = 0)
        scenelabels['bbox3d'] = np.concatenate([scenelabels['bbox3d'], gt['bbox3d'].reshape((1, -1))], axis = 0)
        s += 1
    chosen = sorted(chosen, key = lambda x: x['bbox3d'][2], reverse = True)
    for c in chosen:
        scenevelo, sceneimg = paste(scenevelo, sceneimg, c['velo'], c['img'], c['mask'], c['bbox2d'])
    return scenevelo, sceneimg, scenelabels