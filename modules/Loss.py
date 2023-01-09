from torch import nn
import torch
from torch.nn import SmoothL1Loss
from modules import Config as cfg

class VoxelLoss(nn.Module):

    def __init__(self, a = 1.5, b = 1, eps = 1e-6):
        super().__init__()
        self.a = a
        self.b = b
        self.eps = eps
        self.smoothl1 = SmoothL1Loss()

    def forward(self, pi, ni, gi, gts, score, reg, anchors, anchorsPerLoc):
        # if self.pos is None:
        #     self.pos = torch.empty(score.shape, device = 'cuda')
        #     self.neg = torch.empty(score.shape, device = 'cuda')

        if pi is None:
            clsLoss = -torch.log(1 - score + self.eps).mean()
            return clsLoss, None

        # self.pos[...] = 0
        # self.neg[...] = 1
        # self.pos[pi] = 1
        # self.neg[ni] = 0

        # pos = torch.zeros(score.shape, device = 'cuda')
        # neg = torch.ones(score.shape, device = 'cuda')
        # pos[pi] = 1
        # neg[ni] = 0

        posLoss = -torch.log(score[pi] + self.eps).sum()
        negLoss = -torch.log(1 - score + self.eps)
        sizeSum = negLoss.shape[0] * negLoss.shape[1] * negLoss.shape[2]
        negLoss = negLoss.sum() - negLoss[ni].sum()
        posLoss = posLoss / (pi[0].shape[0] + self.eps)
        negLoss = negLoss / (sizeSum - ni[0].shape[0] + self.eps)
        clsLoss = self.a * posLoss + self.b * negLoss

        if len(pi[0]) == 0:
            return clsLoss, None

        alignedGTs = gts[gi]
        anchors = anchors.reshape((anchors.shape[0], anchors.shape[1], anchorsPerLoc, 7))
        alignedAnchors = anchors[pi]
        d = torch.sqrt(alignedAnchors[:, 3] ** 2 + alignedAnchors[:, 4] ** 2)[:, None]
        targets = torch.empty_like(alignedGTs, device = cfg.device)
        targets[:, [0, 1]] = (alignedGTs[:, [0, 1]] - alignedAnchors[:, [0, 1]]) / d
        targets[:, 2] = (alignedGTs[:, 2] - alignedAnchors[:, 2]) / alignedAnchors[:, 5]
        targets[:, 3:6] = torch.log(alignedGTs[:, 3:6] / alignedAnchors[:, 3:6])
        targets[:, 6] = alignedGTs[:, 6] - alignedAnchors[:, 6]

        reg = reg.reshape((reg.shape[0], reg.shape[1], anchorsPerLoc, 7))[pi]
        regLoss = self.smoothl1(reg, targets)

        return clsLoss, regLoss

class VoxelLoss2(nn.Module):

    def __init__(self, a = 1.5, b = 1, eps = 1e-6):
        super().__init__()
        self.a = a
        self.b = b
        self.eps = eps
        self.smoothl1 = SmoothL1Loss()

    def forward(self, pos, neg, gi, gts, score, reg, anchors, anchorsPerLoc):
        if pos is None:
            clsLoss = -torch.log(1 - score + self.eps).mean()
            return clsLoss, None

        pi = torch.where(pos)

        posLoss = -torch.log(score + self.eps) * pos
        negLoss = -torch.log(1 - score + self.eps) * neg
        posLoss = posLoss.sum() / (len(pi[0]) + self.eps)
        negLoss = negLoss.sum() / (neg.sum() + self.eps)
        clsLoss = self.a * posLoss + self.b * negLoss

        if len(pi[0]) == 0:
            return clsLoss, None

        gi = gi[pi]
        alignedGTs = gts[gi]
        anchors = anchors.reshape((anchors.shape[0], anchors.shape[1], anchorsPerLoc, 7))
        alignedAnchors = anchors[pi]
        d = torch.sqrt(alignedAnchors[:, 3] ** 2 + alignedAnchors[:, 4] ** 2)[:, None]
        targets = torch.empty_like(alignedGTs, device = 'cuda')
        targets[:, [0, 1]] = (alignedGTs[:, [0, 1]] - alignedAnchors[:, [0, 1]]) / d
        targets[:, 2] = (alignedGTs[:, 2] - alignedAnchors[:, 2]) / alignedAnchors[:, 5]
        targets[:, 3:6] = torch.log(alignedGTs[:, 3:6] / alignedAnchors[:, 3:6])
        targets[:, 6] = alignedGTs[:, 6] - alignedAnchors[:, 6]

        reg = reg.reshape((reg.shape[0], reg.shape[1], anchorsPerLoc, 7))[pi]
        regLoss = self.smoothl1(reg, targets)

        return clsLoss, regLoss
