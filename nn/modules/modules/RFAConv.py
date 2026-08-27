from einops import rearrange
import torch.nn as nn
from ultralytics.nn.modules import *

class Bottleneck1(nn.Module):
    """Standard bottleneck."""
    # __init__ 方法：初始化函数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        # 计算隐藏通道数
        c_ = int(c2 * e)  # hidden channels
        # 两个卷积层，分别是输入通道数到隐藏通道数和隐藏通道数到输出通道数的卷积。
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        # 判断是否使用快捷连接，条件是启用快捷连接并且输入通道数等于输出通道数。
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

def autopad(k, p=None, d=1):
    """Pad to 'same' output size (inference mode)."""
    if d > 1:
        k = k + (k - 1) * (d - 1)
    if p is None:
        p = k // 2
    return p

class RFAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None):
        super().__init__()
        # 存储卷积核的尺寸
        self.kernel_size = kernel_size

        # 生成padding
        self.padding=autopad(kernel_size, padding)
        # 生成权重
        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=self.padding, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                   groups=in_channel, bias=False))
        # 生成特征
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=self.padding,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())
        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w
        weighted_data = feature * weighted
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)
        return self.conv(conv_data)

# Bottleneck1的子类
class Bottleneck_RFAConv(Bottleneck1):
    """Standard bottleneck with RFAConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = RFAConv(c_, c2, k[1])

# C3的子类
class C3_RFAConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_RFAConv(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))