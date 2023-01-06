import sys
import time
import numpy as np
import torch
from modules import Pipe, Preprocessing, Loss, Calc
import os
from torch import nn
from torch.autograd import Variable
import pandas as pd

def initWeights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()

class VoxelNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.svfe = Pipe.SVFE()
        self.fcn = Pipe.FCN(128, 128)
        self.cml = Pipe.CML()
        self.rpn = Pipe.RPN()

    @staticmethod
    def reindex(x, idx):
        # input shape: x = (batch * N, 128), idx = (batch * N, 1 + 3)
        res = Variable(torch.Tensor(1, 128, 10, 352, 400).cuda())
        res[idx[:, 0], :, idx[:, 3], idx[:, 1], idx[:, 2]] = x
        return res

    def forward(self, x, idx):
        # x shape = (batch, N, 35, 7), idx shape = (batch * N, 1 + 3), idx[:, 0] is the batch No.
        # idx: corresponding indices to voxels
        x = self.svfe(x)
        x = self.fcn(x)
        # after SVFE&FCN: shape = (batch, N, 35, 128)
        x = torch.max(x, dim = 2)[0]
        # after elementwise max: shape = (batch, N, 1, 128)
        x = torch.squeeze(x, dim = 2).reshape((-1, 128))
        # shape = (batch * N, 128)
        x = self.reindex(x, idx)
        x = self.cml(x)
        x = x.reshape((1, -1, 352, 400))
        score, reg = self.rpn(x)
        return score, reg


dataroot = '../mmdetection3d-master/data/kitti'
if len(sys.argv) > 1:
    dataroot = sys.argv[1]
veloroot = os.path.join(dataroot, 'training/velodyne_reduced')
trainInfoPath = os.path.join(dataroot, 'ImageSets/train.txt')
testInfoPath = os.path.join(dataroot, 'ImageSets/val.txt')
labelroot = os.path.join(dataroot, 'training/label_2')
calibroot = os.path.join(dataroot, 'training/calib')

with open(trainInfoPath, 'r') as f:
    trainSet = f.readlines()
with open(testInfoPath, 'r') as f:
    testSet = f.readlines()

trainX = []
trainY = []
testX = []
testY = []

rangeMin = torch.Tensor(Preprocessing.velorange[:3])
rangeMax = torch.Tensor(Preprocessing.velorange[3:])

for i, s in enumerate(trainSet):
    print(f'\rProcess(train set): {i + 1} / {len(trainSet)}', end = '')
    if s[-1] == '\n':
        s = s[:-1]
    path = os.path.join(veloroot, s + '.bin')
    trainX.append(np.fromfile(path, dtype = 'float32').reshape((-1, 4)))
    path = os.path.join(labelroot, s + '.txt')
    labels = pd.read_csv(path, sep = ' ', index_col = 0, usecols = [0, *[_ for _ in range(8, 15)]])
    labels = labels[labels.index == 'Car'].to_numpy()
    if len(labels) == 0:
        trainY.append((None, None))
        continue
    path = os.path.join(calibroot, s + '.txt')
    with open(path, 'r') as f:
        l = f.readlines()[5][:-1]
        v2c = np.array(l.split(' ')[1:]).astype('float32').reshape((3, 4))
        v2c = np.concatenate([v2c, [[0, 0, 0, 1]]], axis = 0)
        c2v = np.linalg.inv(v2c)
    labels = torch.Tensor(labels)
    c2v = torch.Tensor(c2v)
    Calc.bboxCam2Lidar(labels, c2v, True)
    inRange = torch.all(labels[:, :3].__lt__(rangeMax[None, ...]), dim = 1) & \
              torch.all(labels[:, :3].__ge__(rangeMin[None, ...]), dim = 1)
    labels = labels[inRange].contiguous()
    bevs = Calc.bbox3d2bev(labels)
    trainY.append((labels, bevs))

print()

for i, s in enumerate(testSet):
    print(f'\rProcess(train set): {i + 1} / {len(testSet)}', end = '')
    if s[-1] == '\n':
        s = s[:-1]
    path = os.path.join(veloroot, s + '.bin')
    testX.append(np.fromfile(path, dtype = 'float32').reshape((-1, 4)))
    path = os.path.join(labelroot, s + '.txt')
    labels = pd.read_csv(path, sep=' ', index_col=0, usecols=[0, *[_ for _ in range(8, 15)]])
    labels = labels[labels.index == 'Car'].to_numpy()
    if len(labels) == 0:
        testY.append((None, None))
        continue
    path = os.path.join(calibroot, s + '.txt')
    with open(path, 'r') as f:
        l = f.readlines()[5][:-1]
        v2c = np.array(l.split(' ')[1:]).astype('float32').reshape((3, 4))
        v2c = np.concatenate([v2c, [[0, 0, 0, 1]]], axis=0)
        c2v = np.linalg.inv(v2c)
    labels = torch.Tensor(labels)
    c2v = torch.Tensor(c2v)
    Calc.bboxCam2Lidar(labels, c2v, True)
    inRange = torch.all(labels[:, :3].__lt__(rangeMax[None, ...]), dim = 1) & \
              torch.all(labels[:, :3].__ge__(rangeMin[None, ...]), dim = 1)
    labels = labels[inRange].contiguous()
    bevs = Calc.bbox3d2bev(labels)
    testY.append((labels, bevs))

print()

anchors = Preprocessing.createAnchors(352 // 2, 400 // 2, Preprocessing.velorange, Preprocessing.carsize)
model = VoxelNet()
anchorBevs = Calc.bbox3d2bev(anchors.reshape((-1, 7)))\
    .reshape((176, 200, 2, 4, 2))
# anchorPolygons = Calc.getPolygons(anchorBevs).reshape((176, 200, 2))
criterion = Loss.VoxelLoss()
opt = torch.optim.Adam(model.parameters(), lr = 0.001)
torch.autograd.set_detect_anomaly(True)

model = model.cuda()
anchors = anchors.cuda()
criterion = criterion.cuda()
model.apply(initWeights)

groupTime = 0
forwardTime = [0, 0]
classifyTime = [0, 0]
lossTime = 0

for i, (x, y) in enumerate(zip(trainX, trainY)):
    # shape = (N, 35, 7)
    st = time.process_time()
    voxel, idx = Preprocessing.group(x, Preprocessing.velorange, Preprocessing.voxelsize, 35)
    # voxel_, idx_ = Preprocessing.group_(x, Preprocessing.velorange, Preprocessing.carsize)

    ed = time.process_time()
    groupTime += ed - st
    # shape = (batch, N, 35, 7)
    voxel = voxel[None, :]
    idx = np.concatenate([np.zeros((idx.shape[0], 1)), idx], axis = 1)

    opt.zero_grad()
    st = time.process_time()
    voxel = torch.Tensor(voxel).cuda()
    idx = torch.LongTensor(idx).cuda()
    ed = time.process_time()
    forwardTime[0] += ed - st
    st = time.process_time()
    score, reg = model(voxel, idx)
    ed = time.process_time()
    forwardTime[1] += ed - st
    score = score.squeeze(dim = 0).permute(1, 2, 0)
    reg = reg.squeeze(dim = 0).permute(1, 2, 0)
    if y[0] is not None:
        st = time.process_time()
        # gtPolygons = Calc.getPolygons(y[1])
        # pos, neg, gi = Calc.classifyAnchors_(gtPolygons, y[0][:, [0, 1]], anchorPolygons, Preprocessing.velorange, 0.45, 0.6)
        # print(y[0][gi[torch.where(pos)]])
        pi, ni, gi = Calc.classifyAnchors(y[1], y[0][:, [0, 1]], anchorBevs, Preprocessing.velorange, 0.45, 0.6)
        ed = time.process_time()
        classifyTime[0] += ed - st
        st = time.process_time()
        # pi = pi.cuda()
        # ni = ni.cuda()
        # gi = gi.cuda()
        l = y[0].cuda() # noqa
        ed = time.process_time()
        classifyTime[1] += ed - st
    else:
        pi, ni, gi, l = None, None, None, None

    st = time.process_time()
    clsLoss, regLoss = criterion(pi, ni, gi, l, score, reg, anchors, 2)
    ed = time.process_time()
    lossTime += ed - st
    loss = clsLoss
    if regLoss is not None:
        loss = loss + regLoss
    if (i + 1) % 50 == 0:
        print('\r', groupTime, forwardTime, classifyTime, lossTime)
    print(f'\r{i + 1}/{len(trainSet)}', 'Classification Loss:', clsLoss.item()
          , 'Regression Loss:', 'None' if regLoss is None else regLoss.item(), end = '')
    loss.backward()
    opt.step()

torch.save(model.state_dict(), './checkpoints/epoch1.pkl')