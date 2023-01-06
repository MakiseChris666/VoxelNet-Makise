from shapely.geometry import Polygon
import torch
import numpy as np
from .Extension import cpp

def getRotationMatrices(r):
    rcos = torch.cos(r).reshape((-1, 1))
    rsin = torch.sin(r).reshape((-1, 1))
    rot = torch.concat([rcos, -rsin, rsin, rcos], dim=1).reshape((-1, 2, 2))
    return rot

def bbox3d2bev(bbox3ds):
    """
    Convert 3d bboxes to bev(in format of corner points)
    @param bbox3ds: (N, 7) in xyzlwhr format
    @return: bev in (N, 4, 2)
    """
    assert bbox3ds.ndim == 2 and bbox3ds.shape[1] >= 7
    res = torch.Tensor([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])
    res = torch.tile(res, (bbox3ds.shape[0], 1, 1))
    res = res * bbox3ds[:, None, [3, 4]]
    # res shape = (N, 4, 2)
    r = bbox3ds[:, 6]
    rot = getRotationMatrices(r)
    # rot shape = (N, 2, 2)
    res = res @ rot
    res = res + bbox3ds[:, None, [0, 1]]
    return res

def iou2d(rs1: torch.Tensor, rs2: torch.Tensor):
    """
    Compute pairwise iou between every box in rs1 and rs2
    @param rs1: (N, 4, 2) in corner points format
    @param rs2: (M, 4, 2) in corner points format
    @return: (N, M),
    """
    res = torch.zeros((rs1.shape[0], rs2.shape[0]))
    assert rs1.ndim == 3 and rs2.ndim == 3 and rs1.shape[1:] == (4, 2) and rs2.shape[1:] == (4, 2)
    p1 = []
    p2 = []
    for r1 in rs1:
        p1.append(Polygon(r1))
    for r2 in rs2:
        p2.append(Polygon(r2))
    for i, r1 in enumerate(p1):
        for j, r2 in enumerate(p2):
            overlap = r1.intersection(r2).area
            res[i, j] = overlap / (r1.area + r2.area - overlap)
    return res

def getPolygons(bboxes):
    """
    Get a numpy array of shapely.geometry.Polygons
    @param bboxes: (N, 4, 2)
    @return:
    """
    res = np.empty(bboxes.shape[0], dtype = 'object')
    for i, p in enumerate(bboxes):
        res[i] = Polygon(p)
    return res

def classifyAnchors(gts, gtCenters, anchors, velorange, negThr, posThr):
    l = (velorange[3] - velorange[0]) / anchors.shape[0]
    w = (velorange[4] - velorange[1]) / anchors.shape[1]
    nls = ((gtCenters[:, 0] - velorange[0] - l / 2) / l + 0.5).long()
    nws = ((gtCenters[:, 1] - velorange[1] - w / 2) / w + 0.5).long()
    pi, ni, gi = cpp._classifyAnchors(gts, anchors, nls, nws, negThr, posThr) # noqa
    return pi, ni, gi

def classifyAnchors_(gts, gtCenters, anchors, velorange, negThr, posThr):
    """

    @param gtCenters:
    @param velorange:
    @param gts: Preprocessed by getPolygons
    @param anchors: Preprocessed by getPolygons
    @param negThr:
    @param posThr:
    @return: (pos, neg, gi), gi is the corresponding indices of gt to every positive anchor
    """
    pos = torch.zeros(anchors.shape, dtype = torch.bool, device = 'cuda')
    neg = torch.ones(anchors.shape, dtype = torch.bool, device = 'cuda')
    gi = torch.zeros_like(pos, dtype = torch.int64, device = 'cuda')
    anchorsPerLoc = anchors.shape[2]
    l = (velorange[3] - velorange[0]) / anchors.shape[0]
    w = (velorange[4] - velorange[1]) / anchors.shape[1]
    nls = ((gtCenters[:, 0] - velorange[0] - l / 2) / l + 0.5).long()
    nws = ((gtCenters[:, 1] - velorange[1] - w / 2) / w + 0.5).long()
    anchorArea = anchors[0, 0, 0].area
    for i, gt in enumerate(gts):
        # cx, cy = gt.centroid.xy
        # cx, cy = cx[0], cy[0]
        # nl = int((cx - velorange[0] - l / 2) / l + 0.5)
        # nw = int((cy - velorange[1] - w / 2) / w + 0.5)
        nl = nls[i]
        nw = nws[i]
        gtArea = gt.area
        for z in range(anchorsPerLoc):
            h = 0
            ioull = 0
            while nl + h < anchors.shape[0]:
                inter = anchors[nl + h, nw, z].intersection(gt).area
                iou = inter / (gtArea + anchorArea - inter)
                if iou < negThr and iou <= ioull:
                    break
                if iou >= posThr:
                    pos[nl + h, nw, z] = 1
                    gi[nl + h, nw, z] = i
                    neg[nl + h, nw, z] = 0
                elif iou >= negThr:
                    neg[nl + h, nw, z] = 0
                v = 1
                ioull = iou
                ioul = 0
                while nw + v < anchors.shape[1]:
                    inter = anchors[nl + h, nw + v, z].intersection(gt).area
                    iou = inter / (gtArea + anchorArea - inter)
                    if iou < negThr and iou <= ioul:
                        break
                    if iou >= posThr:
                        pos[nl + h, nw + v, z] = 1
                        gi[nl + h, nw + v, z] = i
                        neg[nl + h, nw + v, z] = 0
                    elif iou >= negThr:
                        neg[nl + h, nw + v, z] = 0
                    v += 1
                    ioul = iou
                v = -1
                ioul = 0
                while nw + v >= 0:
                    inter = anchors[nl + h, nw + v, z].intersection(gt).area
                    iou = inter / (gtArea + anchorArea - inter)
                    if iou < negThr and iou <= ioul:
                        break
                    if iou >= posThr:
                        pos[nl + h, nw + v, z] = 1
                        gi[nl + h, nw + v, z] = i
                        neg[nl + h, nw + v, z] = 0
                    elif iou >= negThr:
                        neg[nl + h, nw + v, z] = 0
                    v -= 1
                    ioul = iou
                h += 1
            h = -1
            ioull = 0
            while nl + h >= 0:
                inter = anchors[nl + h, nw, z].intersection(gt).area
                iou = inter / (gtArea + anchorArea - inter)
                if iou < negThr and iou <= ioull:
                    break
                if iou >= posThr:
                    pos[nl + h, nw, z] = 1
                    gi[nl + h, nw, z] = i
                    neg[nl + h, nw, z] = 0
                elif iou >= negThr:
                    neg[nl + h, nw, z] = 0
                ioull = iou
                ioul = 0
                v = 1
                while nw + v < anchors.shape[1]:
                    inter = anchors[nl + h, nw + v, z].intersection(gt).area
                    iou = inter / (gtArea + anchorArea - inter)
                    if iou < negThr and iou <= ioul:
                        break
                    if iou >= posThr:
                        pos[nl + h, nw + v, z] = 1
                        gi[nl + h, nw + v, z] = i
                        neg[nl + h, nw + v, z] = 0
                    elif iou >= negThr:
                        neg[nl + h, nw + v, z] = 0
                    v += 1
                    ioul = iou
                v = -1
                ioul = 0
                while nw + v >= 0:
                    inter = anchors[nl + h, nw + v, z].intersection(gt).area
                    iou = inter / (gtArea + anchorArea - inter)
                    if iou < negThr and iou <= ioul:
                        break
                    if iou >= posThr:
                        pos[nl + h, nw + v, z] = 1
                        gi[nl + h, nw + v, z] = i
                        neg[nl + h, nw + v, z] = 0
                    elif iou >= negThr:
                        neg[nl + h, nw + v, z] = 0
                    v -= 1
                    ioul = iou
                h -= 1
    return pos, neg, gi

def bboxCam2Lidar(camBoxes, c2v, inplace = False):
    """

    @param inplace:
    @param camBoxes: (N, 7) in 'hwlxyzr'
    @param c2v: (4, 4). Let v2c be calibration matrix 'Tr_velo_to_cam', which is in shape (3, 4),
    than c2v = np.linalg.inv(np.concatenate([v2c, [[0, 0, 0, 1]]], axis = 0))
    @return: (N, 7) in 'xyzlwhr'
    """
    if not inplace:
        t = torch.empty_like(camBoxes)
        t[...] = camBoxes
        camBoxes = t
    xyz = camBoxes[:, 3:6]
    xyz = torch.concat([xyz, torch.ones((camBoxes.shape[0], 1))], dim = 1).T
    xyz = c2v @ xyz
    xyz = xyz.T
    camBoxes[:, 3:6] = camBoxes[:, [2, 1, 0]]
    camBoxes[:, :3] = xyz[:, :3]
    camBoxes[:, 6] = camBoxes[:, 6] - 0.5 * torch.pi
    return camBoxes