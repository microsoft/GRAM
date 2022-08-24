import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterBlock(nn.Module):
    def __init__(self, img_channels, output_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, output_channels, 1, padding=0),
            nn.LeakyReLU(0.2)
        )
    def forward(self, input):
        return self.model(input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)
        # self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        # ret = self.conv(x)
        return ret


class ResidualCoordConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=False, groups=1):
        super().__init__()
        p = kernel_size//2
        self.network = nn.Sequential(
            CoordConv(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=p),
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(planes, planes, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.network.apply(kaiming_leaky_init)

        self.proj = nn.Conv2d(inplanes, planes, 1) if inplanes != planes else None
        self.downsample = downsample

    def forward(self, identity):
        y = self.network(identity)

        if self.downsample: y = nn.functional.avg_pool2d(y, 2)
        if self.downsample: identity = nn.functional.avg_pool2d(identity, 2)
        identity = identity if self.proj is None else self.proj(identity)

        y = (y + identity)/math.sqrt(2)
        return y
