import torch
import torch.nn as nn
import torch.nn.functional as F


class Hardswish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.relu6(x + 3.0) / 6.0


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, leaky=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if leaky else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


class SAM(nn.Module):
    """ Parallel CBAM """
    def __init__(self, in_ch):
        super(SAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
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
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)

        return x


# Copy from yolov5
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c_, c2, k=3, p=1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# Copy from yolov5
class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = nn.Conv2d(c1, c_, kernel_size=1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, kernel_size=1, bias=False)
        self.cv4 = Conv(2 * c_, c2, k=1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
