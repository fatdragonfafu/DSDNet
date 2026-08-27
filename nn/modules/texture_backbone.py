import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
# from ultralytics.nn.modules.Directional_conv import DirectionalConv

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
    
class DirectionalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, direction='vertical'):
        super().__init__()
        self.stride = stride
        # 计算padding，如果未指定则自动计算
        if padding is None:
            padding_v = (1, 0) if stride == 1 else (0, 0)  # stride>1时，第一个卷积不padding
            padding_h = (0, 1) if stride == 1 else (0, 0)
        else:
            padding_v = (padding, 0)
            padding_h = (0, padding)
        
        if direction == 'vertical':
            # 第一个卷积：垂直方向 (3, 1)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 
                                  kernel_size=(3, 1), stride=(stride, 1), padding=padding_v)
            # 第二个卷积：水平方向 (1, 3)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 
                                  kernel_size=(1, 3), stride=1, padding=padding_h)
        elif direction == 'horizontal':
            # 第一个卷积：水平方向 (1, 3)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 
                                  kernel_size=(1, 3), stride=(1, stride), padding=padding_h)
            # 第二个卷积：垂直方向 (3, 1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 
                                  kernel_size=(3, 1), stride=1, padding=padding_v)
        
        # 如果stride>1，需要额外下采样另一个维度，确保与normal分支尺寸一致
        # vertical方向：已经下采样了高度，需要下采样宽度
        # horizontal方向：已经下采样了宽度，需要下采样高度
        if stride > 1:
            if direction == 'vertical':
                # vertical已经下采样了高度，需要下采样宽度
                self.downsample = nn.AvgPool2d(kernel_size=(1, stride), stride=(1, stride))
            else:  # horizontal
                # horizontal已经下采样了宽度，需要下采样高度
                self.downsample = nn.AvgPool2d(kernel_size=(stride, 1), stride=(stride, 1))
        else:
            self.downsample = nn.Identity()
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.downsample(x)  # 补充下采样，确保尺寸一致
        x = self.bn1(x)
        x = self.act(x)
        return x

class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, g=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DirectionalBottleneck(nn.Module):
    """Directional bottleneck with DirectionalConv."""
    def __init__(self,in_channels:int,out_channels:int, shortcut:bool=True, g:int=1, 
        k:Tuple[int,int]=(3,3), e:float=0.5, direction:str='vertical'):
        super().__init__()
        hidden_channels=int(out_channels*e)
        self.cv1=DirectionalConv(in_channels,hidden_channels,1,direction=direction)
        self.cv2=DirectionalConv(hidden_channels,out_channels,1,direction=direction)
        self.add=shortcut and in_channels==out_channels
    
    def forward(self,x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Cased_DirectionalC3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, e: float = 0.5, direction: str = 'vertical', shortcut: bool = True, g: int = 1):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            n: 重复次数
            e: expansion ratio (默认0.5)
            direction: 方向 ('vertical' 或 'horizontal')
            shortcut: 是否使用shortcut (默认True)
            g: groups参数 (默认1)
        
        注意：参数顺序与YAML配置匹配：[out_channels, n, e, direction]
        """
        super().__init__()
        # 参数类型检查和转换
        if isinstance(e, str):
            raise ValueError(f"Cased_DirectionalC3: parameter 'e' received string '{e}' instead of float. "
                           f"Check YAML configuration. Args: in_channels={in_channels}, out_channels={out_channels}, "
                           f"n={n}, e={e}, direction={direction}")
        if not isinstance(e, (int, float)):
            raise ValueError(f"Cased_DirectionalC3: parameter 'e' must be float, got {type(e)}: {e}")
        e = float(e)  # 确保e是float类型
        hidden_channels = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)  # 主分支初始
        self.cv2 = Conv(in_channels, hidden_channels, 1, 1)  # 残差分支
        # 关键修改1: 输出通道从2*c_改为(2+n)*c_
        self.cv3 = Conv((2 + n) * hidden_channels, out_channels, 1)
        # 关键修改2: Sequential改为ModuleList，才能访问每个Bottleneck
        self.m = nn.Sequential(*(DirectionalBottleneck(hidden_channels, hidden_channels, shortcut, g, k=(1, 3), e=1.0, direction=direction) for _ in range(n)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """级联式特征融合的前向传播"""
        # 残差分支：直接保留
        
        # 级联式特征列表：[残差分支, 主分支初始, bottleneck1输出, bottleneck2输出, ...]
        y = [self.cv1(x), self.cv2(x)]
        
        y.extend(m(y[-1]) for m in self.m)
        
        # 关键修改4: 融合所有(2+n)个特征图
        return self.cv3(torch.cat(y, 1))  # concat后: (B, (2+n)*c_, H, W)

class AdaptiveDirectionalConv(nn.Module):
    """
    自适应方向选择的方向性卷积
    
    核心思想：
    - 使用注意力机制自动选择垂直或水平方向
    - 根据输入特征自适应调整方向权重
    - 支持多方向融合（垂直、水平、对角线等）
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 num_directions=4, reduction=16):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            num_directions: 方向数量 (2=垂直+水平, 4=垂直+水平+两个对角线)
            reduction: 注意力机制的压缩比
        """
        super().__init__()
        self.num_directions = num_directions
        self.stride = stride
        
        # 定义多个方向的卷积
        # 方向0: 垂直 (3, 1) -> (1, 3)
        # 方向1: 水平 (1, 3) -> (3, 1)
        # 方向2: 对角线1 (3, 3) 对角线
        # 方向3: 对角线2 (3, 3) 对角线
        
        self.directional_convs = nn.ModuleList()
        
        # 为了保证不同方向分支的输出 H/W 尺寸一致：
        # - 当 stride>1 时，所有分支都 **同时** 下采样 H 和 W（stride=(s,s)）
        # - 这样加权融合时不会出现 (H/2, W) vs (H, W/2) 的尺寸冲突
        padding_v = (1, 0)  # (3,1) 的“same”高度padding
        padding_h = (0, 1)  # (1,3) 的“same”宽度padding
        
        # 方向0: 垂直
        conv_v = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), 
                     stride=(stride, stride), padding=padding_v, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), 
                     stride=1, padding=padding_h, bias=False)
        )
        self.directional_convs.append(conv_v)
        
        # 方向1: 水平
        conv_h = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), 
                     stride=(stride, stride), padding=padding_h, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), 
                     stride=1, padding=padding_v, bias=False)
        )
        self.directional_convs.append(conv_h)
        
        # 如果支持更多方向，添加对角线方向
        if num_directions >= 4:
            # 方向2: 对角线1 (左上到右下)
            conv_d1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                         stride=stride, padding=1, bias=False),
                # 可以添加旋转或特殊处理
            )
            self.directional_convs.append(conv_d1)
            
            # 方向3: 对角线2 (右上到左下)
            conv_d2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                         stride=stride, padding=1, bias=False),
            )
            self.directional_convs.append(conv_d2)
        
        # 自适应方向选择模块 (基于通道注意力)
        self.direction_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, num_directions, 1),
            nn.Softmax(dim=1)
        )
        
        # 下采样层（如果需要）
        # 这里不再做额外 downsample：因为各方向卷积分支已用 stride=(s,s) 完成下采样
        self.downsample = nn.Identity()
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        """
        前向传播：
        1. 计算方向注意力权重
        2. 对每个方向进行卷积
        3. 根据注意力权重加权融合
        """
        B, C, H, W = x.shape
        
        # 计算方向注意力权重 [B, num_directions, 1, 1]
        direction_weights = self.direction_attention(x)  # [B, num_directions, 1, 1]
        
        # 对每个方向进行卷积（各分支输出尺寸一致：约为 ceil(H/stride) x ceil(W/stride)）
        directional_outputs = []
        for i, conv in enumerate(self.directional_convs):
            out = conv(x)
            directional_outputs.append(out)
        
        # 根据注意力权重加权融合
        # direction_weights: [B, num_directions, 1, 1]
        # directional_outputs: List of [B, out_channels, H', W']
        weighted_output = torch.zeros_like(directional_outputs[0])
        for i, out in enumerate(directional_outputs):
            weight = direction_weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
            weighted_output += weight * out
        
        # 不做额外 downsample（避免双重下采样/尺寸不一致）
        weighted_output = self.downsample(weighted_output)
        weighted_output = self.bn(weighted_output)
        weighted_output = self.act(weighted_output)
        
        return weighted_output

class AdaptiveDirectionalBottleneck(nn.Module):
    """Directional bottleneck with AdaptiveDirectionalConv."""
    def __init__(self,in_channels:int,out_channels:int, shortcut:bool=True, g:int=1, 
        k:Tuple[int,int]=(3,3), e:float=0.5):
        super().__init__()
        hidden_channels=int(out_channels*e)
        self.cv1=AdaptiveDirectionalConv(in_channels,hidden_channels,1, num_directions=2, reduction=16)
        self.cv2=AdaptiveDirectionalConv(hidden_channels,out_channels,1, num_directions=2, reduction=16)
        self.add=shortcut and in_channels==out_channels
    
    def forward(self,x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Cased_AdaptiveDirectionalC3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        hidden_channels = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)  # 主分支初始
        self.cv2 = Conv(in_channels, hidden_channels, 1, 1)  # 残差分支
        # 关键修改1: 输出通道从2*c_改为(2+n)*c_
        self.cv3 = Conv((2 + n) * hidden_channels, out_channels, 1)
        # 关键修改2: Sequential改为ModuleList，才能访问每个Bottleneck
        self.m = nn.Sequential(*(AdaptiveDirectionalBottleneck(hidden_channels, hidden_channels, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """级联式特征融合的前向传播"""
        # 残差分支：直接保留
        
        # 级联式特征列表：[残差分支, 主分支初始, bottleneck1输出, bottleneck2输出, ...]
        y = [self.cv1(x), self.cv2(x)]
        
        y.extend(m(y[-1]) for m in self.m)
        
        # 关键修改4: 融合所有(2+n)个特征图
        return self.cv3(torch.cat(y, 1))  # concat后: (B, (2+n)*c_, H, W)



class Cased_C3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, e: float = 0.5, shortcut: bool = True, g: int = 1):
        super().__init__()
       
        hidden_channels = int(out_channels * e) 
        # 确保 cv3 的 in_channels 能被 groups 整除（Conv 默认 g=1，显式传入避免歧义）
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv2 = Conv(in_channels, hidden_channels, 1, 1)
        self.cv3 = Conv((2 + n) * hidden_channels, out_channels, 1, g=1)
        # 关键修改2: Sequential改为ModuleList，才能访问每个Bottleneck
        self.m = nn.ModuleList(
            [Bottleneck(hidden_channels, hidden_channels, shortcut, g, k=(1, 3), e=1.0) for _ in range(max(0, int(n)))]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """级联式特征融合的前向传播"""
        # 残差分支：直接保留
        
        # 级联式特征列表：[残差分支, 主分支初始, bottleneck1输出, bottleneck2输出, ...]
        y = [self.cv1(x), self.cv2(x)]
        
        y.extend(m(y[-1]) for m in self.m)
        
        # 关键修改4: 融合所有(2+n)个特征图
        return self.cv3(torch.cat(y, 1))  # concat后: (B, (2+n)*c_, H, W)
        
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """
        Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv(in_channels * 2, out_channels, 1, 1)

    def forward(self, x1, x2):
        # 确保两个输入的空间尺寸一致
        if x1.shape[-2:] != x2.shape[-2:]:
            # 使用较小的尺寸作为目标尺寸
            target_h = min(x1.shape[-2], x2.shape[-2])
            target_w = min(x1.shape[-1], x2.shape[-1])
            x1 = F.adaptive_avg_pool2d(x1, (target_h, target_w))
            x2 = F.adaptive_avg_pool2d(x2, (target_h, target_w))
        return self.conv(torch.cat((x1, x2), dim=1))






class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )        


            
class DualTextureBackbone(nn.Module):
    def __init__(self, in_channels: int, channels: list = None, depth: int = 2, return_p2: bool = False, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Args:
            in_channels: 输入通道数
            channels: 通道数列表 [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
                      如果为None，使用标准YOLO11n通道数 [64, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024]
            depth: C3k2的重复次数（不再使用depth缩放因子）
            return_p2: 若为 True，返回 [P2, P3, P4, P5]（供 BackboneNeckFusionNeck 等使用）；否则返回 [P3, P4, P5]
            n: 其他参数
            shortcut: 是否使用shortcut
            g: groups
            e: expansion ratio
        """
        super().__init__()
        self.return_p2 = bool(return_p2)
        
        # 初始化通道数列表
        if channels is None:
            channels = [64, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024]
        if len(channels) < 10:
            default_channels = [64, 128, 256, 256, 512, 512, 512, 1024, 1024, 1024]
            channels = channels + default_channels[len(channels):]
        
        c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = channels[:10]
        
        # ========== 分支1：纵向卷积 ==========
        self.vertical_branch = nn.ModuleList(
            [
                Conv(in_channels, c1, 3, 2),#0-P1/2

                DirectionalConv(c1, c2, 3, 2, direction='vertical'),

                Cased_DirectionalC3(c2, c3, depth, e=0.25, direction='vertical'),

                Conv(c3, c4, 3, 2),#3-P3/8

                Cased_DirectionalC3(c4, c5, depth, e=0.25, direction='vertical'),

                Conv(c5, c6, 3, 2),#5-P4/16

                Cased_DirectionalC3(c6, c7, depth, e=0.25, direction='vertical'),

                Conv(c7, c8, 3, 2),#7-P5/32

                Cased_DirectionalC3(c8, c9, depth, e=0.25, direction='vertical'),

                SPPF(c9, c10, 5),#9
                C2PSA(c10, c10),
            ]
        )
        self.normal_branch = nn.ModuleList(
            [
                Conv(in_channels, c1, 3, 2),#0-P1/2
                Conv(c1, c2, 3, 2),#1-P2/4
                C3k2(c2, c3, depth, c3k=False),
                Conv(c3, c4, 3, 2),#3-P3/8
                C3k2(c4, c5, depth, c3k=False),
                Conv(c5, c6, 3, 2),#5-P4/16
                C3k2(c6, c7, depth, c3k=True),
                Conv(c7, c8, 3, 2),#7-P5/32
                C3k2(c8, c9, depth, c3k=True),
                SPPF(c9, c10, 5),#9
                C2PSA(c10, c10),  # 标准yolo11n使用n=2
            ]
        )
        # 融合层的通道数（根据两个分支的实际输出）
        # P3: vertical_branch[4]输出c5通道, normal_branch[4]输出c5通道
        # P4: vertical_branch[6]输出c7通道, normal_branch[6]输出c7通道
        # P5: vertical_branch[10]输出c10通道, normal_branch[10]输出c10通道
        # 使用DynamicFusion替代FeatureFusion，提供更强大的特征融合能力
        # self.fusion_p3 = TripletStyleFusion(c5, c5)  # P3: c5通道
        # self.fusion_p4 = TripletStyleFusion(c7, c7)  # P4: c7通道
        # self.fusion_final = TripletStyleFusion(c10, c10)  # P5: c10通道


        self.fusion_p2 = FeatureFusion(c3, c3)  # P2: c3通道（索引2，1/4 下采样）
        self.fusion_p3 = FeatureFusion(c5, c5)  # P3: c5通道
        self.fusion_p4 = FeatureFusion(c7, c7)  # P4: c7通道
        self.fusion_final = FeatureFusion(c10, c10)  # P5: c10通道

        # 保存输出通道数，用于返回
        self.out_channels_p2 = c3  # P2: 融合后输出c3通道
        self.out_channels_p3 = c5  # P3: 融合后输出c5通道
        self.out_channels_p4 = c7  # P4: 融合后输出c7通道
        self.out_channels_p5 = c10  # P5: 融合后输出c10通道

    def forward(self, x):
        """
        前向传播 - 双分支融合
        返回：
            x3: P3特征 (融合后的c5通道)
            x4: P4特征 (融合后的c7通道)
            x5: P5特征 (融合后的c10通道)
        """
        v_features = []
        h_features = []
        
        # 同时运行两个分支
        v_feat = x
        h_feat = x
        for v_layer, h_layer in zip(self.vertical_branch, self.normal_branch):
            v_feat = v_layer(v_feat)
            h_feat = h_layer(h_feat)
            v_features.append(v_feat)
            h_features.append(h_feat)
        
        # 在关键位置进行特征融合
        # P2: 索引2之后（1/4 下采样），供 BackboneNeckFusionNeck 等使用
        x2 = self.fusion_p2(v_features[2], h_features[2])
        # P3: 索引4之后（vertical_branch[4]和normal_branch[4]）
        x3 = self.fusion_p3(v_features[4], h_features[4])
        # P4: 索引6之后（vertical_branch[6]和normal_branch[6]）
        x4 = self.fusion_p4(v_features[6], h_features[6])
        # P5: 索引10之后（vertical_branch[10]和normal_branch[10]）
        x5 = self.fusion_final(v_features[10], h_features[10])

        if self.return_p2:
            return [x2, x3, x4, x5]
        return [x3, x4, x5]
        


# class TextureBackboneAdapter(nn.Module):
#     """
#     Texture Backbone适配器，使其可以在YOLO11的YAML配置中使用
#     这个适配器将DualTextureBackbone的输出格式转换为YOLO11期望的格式
    
#     返回格式：[P3, P4, P5]，可以使用Index模块来提取特定特征图
#     """
#     def __init__(self, c1, channels: list = None, depth: int = 2,  *args, **kwargs):
#         """
#         Args:
#             c1: 输入通道数
#             channels: 通道数列表 [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
#                       如果为None，使用标准YOLO11n通道数（已由parse_model缩放）
#             depth: C3k2的重复次数
#         """
#         super().__init__()
            
#         self.backbone = DualTextureBackbone(in_channels=c1, channels=channels, depth=depth)
#         # 保存输出通道数，用于返回给parse_model
#         self.out_channels_p3 = self.backbone.out_channels_p3
#         self.out_channels_p4 = self.backbone.out_channels_p4
#         self.out_channels_p5 = self.backbone.out_channels_p5
        
#     def forward(self, x):
#         """
#         前向传播，返回多个特征图
#         返回: [P3, P4, P5] 特征图列表
#         """
#         results = self.backbone(x)
#         # 返回 [P3, P4, P5] 列表，可以使用Index模块提取
#         return results

if __name__ == "__main__":
    # 测试模型
    model = DualTextureBackbone(in_channels=3)
    
    # 创建测试输入
    x = torch.randn(1, 3, 640, 640)
    
    # 前向传播
    backbone_results = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"P3特征图形状: {backbone_results[0].shape}")
    print(f"P4特征图形状: {backbone_results[1].shape}")
    print(f"最终特征图形状: {backbone_results[2].shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
