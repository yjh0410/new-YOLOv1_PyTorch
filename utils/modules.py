import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1, leakyReLU=False):
        super(Conv2d, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True) if leakyReLU else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class SAM(nn.Module):
    """ Parallel CBAM """
    def __init__(self, in_ch):
        super(SAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.Sigmoid()           
        )

    def forward(self, x):
        """ Spatial Attention Module """
        x_attention = self.conv(x)

        return x * x_attention


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self, in_ch, out_ch):
        super(SPP, self).__init__()
        self.fuse_conv = Conv2d(in_ch * 4, out_ch, 1, leakyReLU=True)

    def forward(self, x):
        x_1 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = F.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)

        return self.fuse_conv(x)
