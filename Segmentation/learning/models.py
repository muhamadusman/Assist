"""Adapted from https://github.com/jvanvugt/pytorch-unet (MIT license)"""
import math
from typing import Optional, OrderedDict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from learning.utils import tensor_center_crop


class UNet(nn.Module):
    def __init__(self,
                 output_size: Optional[Union[int, Tuple[int, int]]] = None,
                 in_channels: int = 1,
                 n_classes: int = 2,
                 depth: int = 6,
                 wf: int = 6,
                 padding='valid',
                 instance_norm: bool = True,
                 up_mode: str = 'upsample',
                 leaky: bool = True,
                 output_all_levels: bool = True):
        """Adapted from https://github.com/jvanvugt/pytorch-unet
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Args:
            in_channels: number of input channels
            n_classes: number of output channels
            depth: depth of the network
            wf: number of filters in the first layer is 2**wf
            padding: controls the amount of padding applied to the input for each conv2d. It can be either a string 
                     {'valid', 'same'}  or a tuple of ints giving the amount of implicit padding applied on both sides.
            instance_norm: Use InstanceNorm after layers with an activation function
            up_mode: one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        assert padding in ('valid', 'same')

        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.leaky = leaky
        self.down_path = nn.ModuleList()
        self.output_all_levels = output_all_levels
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i), padding, instance_norm, leaky=self.leaky))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        self.output_layers = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, instance_norm, leaky))
            if i == 0:  # last layer do not need upsampling
                self.output_layers.append(nn.Conv2d(2 ** (wf + i), n_classes, kernel_size=1, bias=False))
            else:
                self.output_layers.append(nn.Sequential(nn.Conv2d(2 ** (wf + i), n_classes, kernel_size=1, bias=False),
                                                        nn.Upsample(scale_factor=2**i, mode='bilinear')))
            prev_channels = 2 ** (wf + i)

        if output_size is not None:
            self.output_size = output_size
            self.input_size = self.get_minimal_input_size(self.output_size)
            self.fix_size = True
        else:
            self.output_size = None
            self.input_size = None
            self.fix_size = False

    def forward(self, x):
        """If output_all_levels is true, outputs will be ordered from highest to lowest resolution"""
        if self.fix_size:
            assert x.shape[2:] == self.input_size

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        outputs = []
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
            outputs.append(self.output_layers[i](x))

        out = []
        output_size = self.output_size
        for output in reversed(outputs):
            if output_size is None:
                output_size = tuple(output.shape[2:])
            out.append(tensor_center_crop(output, target_size=output_size))

        if not self.output_all_levels:
            out = out[0]
        return out

    def get_minimal_input_size(self, wanted_output_size: Union[int, Tuple[int, int]]):
        if self.depth == 5:
            multiplier = 16
            offset = 188
        elif self.depth == 6:
            multiplier = 32
            offset = 380
        else:
            raise NotImplementedError(f'get_minimal_input_size not implemented for depth {self.depth}')

        if self.padding == 'same':
            offset = 0

        if isinstance(wanted_output_size, int):
            factor = int(math.ceil((wanted_output_size - 4) / multiplier))
            return offset + factor*multiplier
        else:
            factors = tuple([int(math.ceil((wos - 4) / multiplier)) for wos in wanted_output_size])
            return tuple([offset + factor*multiplier for factor in factors])

    def load_from_lightning(self, checkpoint):
        state_dict = OrderedDict()
        for name, weights in checkpoint['state_dict'].items():
            if name.startswith('model'):
                state_dict[name.replace('model.', '', 1)] = weights
        self.load_state_dict(state_dict)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding: int, instance_norm: bool, leaky: bool):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=padding))
        if leaky:
            block.append(nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        else:
            block.append(nn.ReLU())
        if instance_norm:
            block.append(nn.InstanceNorm2d(out_size, affine=True))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=padding))
        if leaky:
            block.append(nn.LeakyReLU(negative_slope=1e-2, inplace=True))
        else:
            block.append(nn.ReLU())
        if instance_norm:
            block.append(nn.InstanceNorm2d(out_size, affine=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, instance_norm, leaky):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, instance_norm, leaky)

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = tensor_center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


if __name__ == '__main__':
    model = UNet(output_size=(256, 256))
    print(model.input_size)
    model = UNet(depth=6, padding='same')
    model.output_all_levels = False
    for factor in range(60):
        in_size = 256 + factor*32
        input = torch.zeros((1, 1, in_size, in_size))
        output = model(input)
        try:
            output = model(input)
        except Exception as e:
            print(f'input size {in_size} crashing')
            continue
        print(f'input size {in_size}, output size {output.size(3)}, diff {in_size - output.size(3)}')
