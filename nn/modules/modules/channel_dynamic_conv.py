import torch
import torch.nn as nn
import torch.nn.functional as F
import platform
import sys
import os
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from modules.conv import Conv
from modules.RFAConv import RFAConv
def autopad(k, p=None, d=1):  
    """Pad to 'same' output size (inference mode)."""
    if d > 1:
        k = k + (k - 1) * (d - 1)
    if p is None:
        p = k // 2
    return p

class Channel_Dynamic_Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=None,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(Channel_Dynamic_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.num_branchs=2
        self.channel_weight=nn.Parameter(torch.randn(self.num_branchs,in_channels))
        padding_=autopad(kernel_size, padding, dilation)

        self.branch1=Conv(self.in_channels,self.out_channels//self.num_branchs,self.kernel_size,self.stride,padding_)
        self.branch2=RFAConv(self.in_channels,self.out_channels//self.num_branchs,self.kernel_size,self.stride, padding_)

    def forward(self, x):
        B,C,H,W=x.shape

        weights=F.softmax(self.channel_weight,dim=0)
        
        split_features=[]
        for i in range(self.num_branchs):
            mask=weights[i].view(1,C,1,1)
            split_features.append(x*mask)

        b1=self.branch1(split_features[0])
        b2=self.branch2(split_features[1])
        #b3=self.branch3(split_features[0])
        bt=torch.cat([b1,b2],dim=1)

        return bt


if  __name__ == '__main__':
    x=torch.randn(1,3,640,640).to("cuda")
    #model=Channel_Dynamic_Conv(3,128, 3, 2).to("cuda")
    model=Conv(3,128, 3, 2).to("cuda")
    y=model(x)
    print(y.shape)

