from torch import nn
import torch.nn.functional as f
import torch

class FCN(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()
        self.fc = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        # input shape = (batch, h, w, c)
        x = f.relu(self.fc(x))
        x = x.permute(0, 3, 1, 2)
        x = self.bn(x)
        # after BN: (batch, c, h, w)
        return x.permute(0, 2, 3, 1)

class CRB3d(nn.Module):

    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, k, s, p)
        self.bn = nn.BatchNorm3d(cout)

    def forward(self, x):
        x = f.relu(self.conv(x))
        return self.bn(x)

class VFE(nn.Module):

    def __init__(self, cin, cout, sampleNum):
        super().__init__()
        self.fcn = FCN(cin, cout)
        self.sampleNum = sampleNum

    def forward(self, x):
        # input shape = (batch, N, 35, cin)
        x = self.fcn(x)
        # shape = (batch, N, 35, cout)
        s = torch.max(x, dim = 2, keepdim = True)[0].repeat(1, 1, self.sampleNum, 1)
        # concat on channels
        return torch.concat([x, s], dim = -1)

class SVFE(nn.Module):

    def __init__(self, sampleNum = 35):
        super().__init__()
        self.vfe1 = VFE(7, 16, sampleNum)
        self.vfe2 = VFE(32, 64, sampleNum)

    def forward(self, x):
        x = self.vfe1(x)
        return self.vfe2(x)

class CML(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = CRB3d(128, 64, 3, (2, 1, 1), (1, 1, 1))
        self.conv2 = CRB3d(64, 64, 3, 1, (0, 1, 1))
        self.conv3 = CRB3d(64, 64, 3, (2, 1, 1), 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class CBR2d(nn.Module):

    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = f.relu(self.conv(x))
        return self.bn(x)

class DeCBR2d(nn.Module):

    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(cin, cout, k, s, p)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.deconv(x)
        return self.bn(x)

class RPN(nn.Module):

    def __init__(self):
        super().__init__()
        self.blk1 = nn.Sequential(
            CBR2d(128, 128, 3, 2, 1),
            *[CBR2d(128, 128, 3, 1, 1) for _ in range(3)]
        )
        self.blk2 = nn.Sequential(
            CBR2d(128, 128, 3, 2, 1),
            *[CBR2d(128, 128, 3, 1, 1) for _ in range(5)]
        )
        self.blk3 = nn.Sequential(
            CBR2d(128, 256, 3, 2, 1),
            *[CBR2d(256, 256, 3, 1, 1) for _ in range(5)]
        )
        self.deconv1 = DeCBR2d(128, 256, 3, 1, 1)
        self.deconv2 = DeCBR2d(128, 256, 2, 2, 0)
        self.deconv3 = DeCBR2d(256, 256, 4, 4, 0)
        self.cls = nn.Conv2d(768, 2, 1, 1, 0)
        self.reg = nn.Conv2d(768, 14, 1, 1, 0)

    def forward(self, x):
        x1 = self.blk1(x)
        x2 = self.blk2(x1)
        x3 = self.blk3(x2)
        dx1 = self.deconv1(x1)
        dx2 = self.deconv2(x2)
        dx3 = self.deconv3(x3)
        x = torch.concat([dx1, dx2, dx3], dim = 1)
        return torch.sigmoid(self.cls(x)), self.reg(x)
