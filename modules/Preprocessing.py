import numpy as np
import torch
from .Extension import cpp

def crop(pcd, range):
    low = np.array(range[0:3])
    high = np.array(range[3:6])
    roi = pcd[:, :3]
    f = np.all((low <= roi) & (roi < high), axis = 1)
    return pcd[f]

def group(pcd, range, size, samplesPerVoxel):
    # np.random.shuffle(pcd)
    pts = pcd[:, :3]
    low = np.array(range[0:3])
    idx = ((pts - low) / size).astype('int32')
    voxel, uidx, vcnt = cpp._group(pcd, idx, samplesPerVoxel) # noqa
    center = voxel[..., :3].sum(axis = 1) / vcnt[:, None]
    voxel[..., 3:6] = voxel[..., :3] - center[:, None, :]
    zero = (voxel[..., :3] == 0).all(axis = 2)
    zero = np.where(zero)
    voxel[zero[0], zero[1], 3:6] = 0
    return voxel, np.array(uidx).T

def group_(pcd, range, size):
    """

    @param pcd:
    @param range:
    @param size:
    @return: (voxel, indices), voxel in (N, 35, 7), indices in (N, 3)
    """
    # np.random.shuffle(pcd)
    pts = pcd[:, :3]
    low = np.array(range[0:3])
    idx = ((pts - low) / size).astype('int32')
    uidx, inv = np.unique(idx, axis = 0, return_inverse = True)
    voxel = np.zeros((len(uidx), 35, 7))
    vcnt = np.zeros(len(uidx), dtype = 'int32')
    for i, p in zip(inv, pcd):
        if vcnt[i] == 35:
            continue
        voxel[i, vcnt[i], [0, 1, 2, 6]] = p
        vcnt[i] += 1
    center = voxel[..., :3].sum(axis = 1) / vcnt[:, None]
    voxel[..., 3:6] = voxel[..., :3] - center[:, None, :]
    zero = (voxel[..., :3] == 0).all(axis = 2)
    zero = np.where(zero)
    voxel[zero[0], zero[1], 3:6] = 0
    return voxel, uidx

def createAnchors(l, w, range, size):
    """

    @param l:
    @param w:
    @param range:
    @param size:
    @return: (l, w, 14)
    """
    ls = (range[3] - range[0]) / l
    ws = (range[4] - range[1]) / w
    x = torch.linspace(range[0] + ls / 2, range[3] - ls / 2, l)
    y = torch.linspace(range[1] + ws / 2, range[4] - ws / 2, w)
    x, y = torch.meshgrid(x, y, indexing = 'ij')
    g = torch.concat([x[..., None], y[..., None]], dim = 2)
    size = torch.Tensor(size)
    size = torch.tile(size, (l, w, 1))
    z = torch.empty((l, w, 1))
    z[...] = -1
    t = torch.zeros((l, w, 1))
    t2 = torch.empty((l, w, 1))
    t2[...] = torch.pi / 2
    anchors = torch.concat([g, z, size, t], dim = 2)
    anchors2 = torch.concat([g, z, size, t2], dim = 2)
    return torch.concat([anchors, anchors2], dim = 2)

velorange = [0, -40, -3, 70.4, 40, 1]
voxelsize = [0.4, 0.2, 0.2]
carsize = [3.9, 1.6, 1.56]
