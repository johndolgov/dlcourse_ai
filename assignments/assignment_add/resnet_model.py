import torch.nn as nn
import torch
from typing import TypeVar, List, Tuple

Vector = TypeVar('Vector', List[int], Tuple[int, int, int, int])
Tensor = TypeVar('Tensor', torch.FloatTensor, torch.cuda.FloatTensor)
shortcut_type = TypeVar('shortcut_type', type(None), nn.modules.container.Sequential)


class BasicBlock(nn.Module):

    """
    Basic block for ResNet model (resnet18, resnet34).
    Basic block contains 2 Convolution layers with ReLU, BatchNorms and optional projection shortcut.

    :param in_filters (int): number  of input filters
    :param out_filters (int): number of output filters
    :param stride (int): stride for Convolution layers
    :param padding (int): padding for Convolution layers
    :param projection_shortcut (None or nn.Sequential): module for projection shortcut

    """

    def __init__(self, in_filters: int, out_filters: int, stride: int = 1,
                 padding: int = 1, projection_shortcut: shortcut_type = None):
        super(BasicBlock, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
                                kernel_size=3, stride=stride, padding=padding, dilation=1)
        self.bn_1 = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=out_filters, out_channels=out_filters,
                                kernel_size=3, stride=1, padding=padding, dilation=1)
        self.bn_2 = nn.BatchNorm2d(out_filters)
        self.projection_shortcut = projection_shortcut

    def forward(self, x: Tensor) -> Tensor:

        """
        Forward pass implementation for Basic block

        :param x (Tensor): Tensor which come as input to Basic block
        :return: Tensor after forward pass
        """

        shortcut = x

        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)

        if self.projection_shortcut:
            shortcut = self.projection_shortcut(shortcut)

        out += shortcut
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):

    """
    Bottleneck block for ResNet model (resnet50, resnet101, resnet152).
    Bottleneck block contains 3 Convolution layers with ReLU, BatchNorms and optional projection shortcut.

    :param in_filters (int): number  of input filters
    :param out_filters (int): number of output filters
    :param stride (int): stride for Convolution layers
    :param padding (int): padding for Convolution layers
    :param projection_shortcut (None or nn.Sequential): module for projection shortcut

    """

    def __init__(self, in_filters: int, out_filters: int, stride: int =1, padding: int =1, projection_shortcut=None):
        super(BottleneckBlock, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
                                kernel_size=1, stride=stride, padding=0, dilation=1)
        self.bn_1 = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=out_filters, out_channels=out_filters,
                                kernel_size=3, stride=1, padding=padding, dilation=1)
        self.bn_2 = nn.BatchNorm2d(out_filters)
        self.conv_3 = nn.Conv2d(in_channels=out_filters, out_channels=4*out_filters,
                                kernel_size=1, stride=1, padding=0, dilation=1)
        self.bn_3 = nn.BatchNorm2d(4*out_filters)
        self.projection_shortcut = projection_shortcut

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass implementation for Bottleneck block

        :param x (Tensor): Tensor which come as input to Basic block
        :return: Tensor after forward pass
        """
        shortcut = x

        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu(out)
        out = self.conv_3(out)
        out = self.bn_3(out)

        if self.projection_shortcut:
            shortcut = self.projection_shortcut(shortcut)

        out += shortcut
        out = self.relu(out)

        return out


Block = TypeVar('Block', BasicBlock, BottleneckBlock)
ResBlock = List[Block]


class Flattener(nn.Module):
    """
    Class Flattener which flat 4d tensor for Fully Connected layer

    """
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass implementation for class Flattener

        :param x (Tensor): Tensor which come as input to  Flattener block
        :return: Flat tensor
        """
        batch_size, *_ = x.shape
        return x.view(batch_size, -1)


class ResNet(nn.Module):
    """
    ResNet class which implement Residual Network from https://arxiv.org/pdf/1512.03385.pdf


    :param block_sizes (Vector): List or Tuple of 4 integer numbers, which
    :param in_channels (int): channels number of input picture
    :param output_layer (int): number of output classes
    :param bottleneck (bool): bottleneck indicator. If True, resnet uses Bottleneck Block, otherwise Basic Block
    """

    def __init__(self, block_sizes: Vector, in_channels: int, output_layer: int, bottleneck: bool = False):
        super(ResNet, self).__init__()

        if not isinstance(block_sizes, (List, Tuple)):
            raise TypeError('block_sizes should be list or tuple')

        if len(block_sizes) != 4:
            raise ValueError('block_size should be length of 4')

        if not isinstance(in_channels, int):
            raise TypeError('in_channels should be int type')

        if not isinstance(output_layer, int):
            raise TypeError('output_layer should be int type')

        if not isinstance(bottleneck, bool):
            raise TypeError('bottleneck should be bool type')

        self.block_sizes = block_sizes
        self.output_layer = output_layer
        self.bottleneck = bottleneck
        self.current_filter_size = 64

        self._conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.current_filter_size,
                                kernel_size=7, stride=2, padding=3, bias=False)

        self._bn1 = nn.BatchNorm2d(self.current_filter_size)
        self._relu = nn.ReLU(inplace=True)

        self._maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._block1 = self.layer(block_size=block_sizes[0], bottleneck=bottleneck,
                                  is_projection_shortcut=False, out_channels=64, stride=1, padding=1)

        self._block2 = self.layer(block_size=block_sizes[1], bottleneck=bottleneck,
                                  is_projection_shortcut=True, out_channels=128, stride=2, padding=1)

        self._block3 = self.layer(block_size=block_sizes[2], bottleneck=bottleneck,
                                  is_projection_shortcut=True, out_channels=256, stride=2, padding=1)

        self._block4 = self.layer(block_size=block_sizes[3], bottleneck=bottleneck,
                                  is_projection_shortcut=True, out_channels=512, stride=2, padding=1)

        self._avpool = nn.AdaptiveAvgPool2d((1, 1))
        self._flat = Flattener()
        self._fc = nn.Linear(in_features=512*(int(self.current_filter_size//512)), out_features=output_layer)

    def forward(self, x: Tensor) -> Tensor:

        """
        Forward pass implementation for class ResNet

        :param x (Tensor): Tensor which come as input to  ResNet block
        :return: Tensor after ResNet model
        """

        out = self._conv1(x)
        out = self._bn1(out)
        out = self._relu(out)
        out = self._maxpool(out)

        out = self._block1(out)
        out = self._block2(out)
        out = self._block3(out)
        out = self._block4(out)

        out = self._avpool(out)
        out = self._flat(out)
        out = self._fc(out)

        return out

    def layer(self, block_size: int, bottleneck: bool, is_projection_shortcut: bool, out_channels: int, stride: int, padding: int):

        """
        Function which creates Bottleneck or Basic blocks according to block size

        :param block_size (int): number of blocks
        :param bottleneck (bool): bottleneck indicator. If True, resnet uses Bottleneck Block, otherwise Basic Block
        :param is_projection_shortcut (bool):projection_shortcut indicator. If True, projection shortcut will be used for this block
        :param out_channels (int): number of channels after forward pass though block
        :param stride (int): stride for Convolutional layer
        :param padding (int): padding for Convolutional layer
        :return: Basic or Bottleneck blocks
        """

        residual_block = nn.Sequential()
        if not bottleneck:
            if is_projection_shortcut or self.current_filter_size != out_channels:

                # Variant (B) in paper

                projection_shortcut = nn.Sequential(nn.Conv2d(in_channels=self.current_filter_size,
                                                              out_channels=out_channels, kernel_size=1, stride=stride),
                                                    nn.BatchNorm2d(out_channels)
                                                    )
            else:
                projection_shortcut = None

            for i in range(block_size):
                residual_block.add_module(f'Basic_block_{self.current_filter_size}x{out_channels}_{i}',
                                          BasicBlock(in_filters=self.current_filter_size, out_filters=out_channels,
                                                     stride=stride, padding=padding,
                                                     projection_shortcut=projection_shortcut))

                self.current_filter_size = out_channels
                stride = 1
                padding = 1
                projection_shortcut = None

            return residual_block

        else:

            # Variant (B) in paper

            if is_projection_shortcut or self.current_filter_size != 4*out_channels:
                projection_shortcut = nn.Sequential(nn.Conv2d(in_channels=self.current_filter_size,
                                                              out_channels=4*out_channels, kernel_size=1, stride=stride),
                                                    nn.BatchNorm2d(4*out_channels)
                                                    )
            else:
                projection_shortcut = None

            for i in range(block_size):
                residual_block.add_module(f'Bottleneck_block_{self.current_filter_size}x{out_channels}_{i}',
                                          BottleneckBlock(in_filters=self.current_filter_size, out_filters=out_channels,
                                                          stride=stride, padding=padding,
                                                          projection_shortcut=projection_shortcut))
                self.current_filter_size = 4*out_channels
                stride = 1
                padding = 1
                projection_shortcut = None

            return residual_block





