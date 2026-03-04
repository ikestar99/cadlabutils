#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@origin: https://github.com/ogliko/patchseq-autorecon
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


__doc__ = """
Symmetric 3d U-Net for multiclass segmentation implemented in PyTorch

(Optional)
Factorized 3D convolution, Extra residual connections

Nicholas Turner <nturner@cs.princeton.edu>, 2017
Based on an architecture by Kisuk Lee <kisuklee@mit.edu>, 2017
"""


# Global switches
FACTORIZE = False
RESIDUAL = True
BN = True

# Number of feature maps
NFEATURES = [16, 16, 64, 128, 80, 96]

# Filter size
SIZES = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]

# In/out filter & stride size
IO_SIZE = (3, 3, 3)
IO_STRIDE = (1, 1, 1)


def pad_size(ks, mode):
    assert mode in ["valid", "same", "full"]

    if mode == "valid":
        return (0, 0, 0)
    elif mode == "same":
        assert all([x % 2 for x in ks])
        return tuple(x // 2 for x in ks)
    elif mode == "full":
        return tuple(x - 1 for x in ks)


class _Conv(nn.Module):
    """ Bare-bones 3D convolution module w/ MSRA init """

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            st: tuple,
            pd: tuple,
            bias: bool = True
    ):
        super(_Conv, self).__init__()
        self.conv = nn.Conv3d(d_in, d_out, ks, st, pd, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(
            self,
            x: torch.tensor
    ):
        return self.conv(x)


class _FactConv(nn.Module):
    """ Factorized 3D convolution using Conv"""

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            st: tuple,
            pd: tuple,
            bias: bool = True
    ):
        super(_FactConv, self).__init__()
        self.factor = None
        if ks[0] > 1:
            self.factor = _Conv(
                d_in, d_out, ks=(1, ks[1], ks[2]), st=(1, st[1], st[2]),
                pd=(0, pd[1], pd[2]), bias=False)
            ks = (ks[0], 1, 1)
            st = (st[0], 1, 1)
            pd = (pd[0], 0, 0)

        self.conv = _Conv(d_in, d_out, ks, st, pd, bias)

    def forward(
            self,
            x: torch.tensor
    ):
        out = (
            self.conv(x) if self.factor is None else self.conv(self.factor(x)))
        return out


class _ConvT(nn.Module):
    """ Bare Bones 3D ConvTranspose module w/ MSRA init """

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            st: tuple,
            pd: tuple = (0, 0, 0),
            bias: bool = True
    ):
        super(_ConvT, self).__init__()
        self.conv = nn.ConvTranspose3d(d_in, d_out, ks, st, pd, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(
            self,
            x: torch.tensor
    ):
        return self.conv(x)


class _Upsample2D(nn.Module):

    def __init__(
            self,
            scale_factor: int,
            mode: str = "nearest"
    ):
        super(_Upsample2D, self).__init__()
        self.scale_factor = scale_factor
        self.upsample = nn.Upsample(scale_factor=2, mode=mode)

    def forward(
            self,
            x: torch.tensor
    ):
        # upsample in all dimensions, and undo the z upsampling
        return self.upsample(x)[:, :, ::self.scale_factor, :, :]


class _FactConvT(nn.Module):
    """Factorized 3d ConvTranspose using ConvT"""

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            st: tuple,
            pd: tuple = (0, 0, 0),
            bias: bool = True
    ):
        super(_FactConvT, self).__init__()
        self.factor = None
        if ks[0] > 1:
            self.factor = _ConvT(
                d_in, d_out, ks=(2, ks[1], ks[2]), st=(1, st[1], st[2]),
                pd=(0, pd[1], pd[2]), bias=False)
            ks = (ks[0], 1, 1)
            st = (st[0], 1, 1)
            pd = (pd[0], 0, 0)

        self.conv = _ConvT(d_in, d_out, ks, st, pd, bias)

    def forward(
            self,
            x: torch.tensor
    ):
        return self.conv(self.factor(x) if self.factor is not None else x)


class _AConv(nn.Module):
    """ Single convolution module """

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            st: tuple = (1, 1, 1),
            activation=F.elu,
            fact: bool = FACTORIZE
    ):
        super(_AConv, self).__init__()
        pd = pad_size(ks, "same")

        conv_constr = _FactConv if fact else _Conv
        self.conv = conv_constr(d_in, d_out, ks, st, pd, bias=True)
        self.activation = activation

    def forward(
            self,
            x: torch.tensor
    ):
        return self.activation(self.conv(x))


class _ConvMod(nn.Module):
    """ Convolution "module" """

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            activation=F.elu,
            fact: bool = FACTORIZE,
            resid: bool = RESIDUAL,
            bn: bool = BN,
            momentum: float = 0.5
    ):
        super(_ConvMod, self).__init__()
        st = (1, 1, 1)
        pd = pad_size(ks, "same")
        conv_constr = _FactConv if fact else _Conv
        bias = not bn

        self.resid = resid
        self.bn = bn
        self.activation = activation

        first_pd = pad_size((1, ks[1], ks[2]), "same")
        self.conv1 = conv_constr(
            d_in, d_out, (1, ks[1], ks[2]), st, first_pd, bias)
        self.conv2 = conv_constr(d_out, d_out, ks, st, pd, bias)
        self.conv3 = conv_constr(d_out, d_out, ks, st, pd, bias)
        if self.bn:
            self.bn1 = nn.BatchNorm3d(d_out, momentum=momentum)
            self.bn2 = nn.BatchNorm3d(d_out, momentum=momentum)
            self.bn3 = nn.BatchNorm3d(d_out, momentum=momentum)

    def forward(
            self,
            x: torch.tensor
    ):
        out1 = self.activation(
            self.bn1(self.conv1(x)) if self.bn else self.conv1(x))
        out2 = self.activation(
            self.bn2(self.conv2(out1)) if self.bn else self.conv2(out1))

        out3 = (self.conv3(out2) + out1) if self.resid else self.conv3(out2)
        out3 = self.activation(self.bn3(out3) if self.bn else out3)
        return out3


class _ConvTMod(nn.Module):
    """ Transposed Convolution "module" """

    def __init__(
            self,
            d_in: int,
            d_out: int,
            ks: tuple,
            up: tuple = (2, 2, 2),
            activation=F.elu,
            fact: bool = FACTORIZE,
            resid: bool = RESIDUAL,
            bn: bool = BN,
            momentum: float = 0.5
    ):
        super(_ConvTMod, self).__init__()

        # ConvT constructor
        convt_constr = _FactConvT if fact else _ConvT
        self.bn = bn
        self.activation = activation
        bias = not bn

        self.convt = convt_constr(d_in, d_out, ks=up, st=up, bias=bias)
        self.convmod = _ConvMod(d_out, d_out, ks, fact=fact, resid=resid, bn=bn)
        if bn:
            self.bn1 = nn.BatchNorm3d(d_out, momentum=momentum)

    def forward(
            self,
            x: torch.tensor,
            skip: torch.tensor
    ):
        out = self.convt(x) + skip
        out = self.convmod(self.activation(self.bn1(out) if self.bn else out))
        return out


class _OutputModule(nn.Module):
    """ Hidden representation -> Output module """

    def __init__(
            self,
            d_in: int,
            outspec: OrderedDict,
            ks: tuple = IO_SIZE,
            st: tuple = IO_STRIDE
    ):
        super(_OutputModule, self).__init__()
        pd = pad_size(ks, mode="same")
        self.output_layers = []
        # self.output_layers = list(outspec.keys())
        for name, d_out in outspec.items():
            setattr(self, name, _Conv(
                d_in, d_out, ks, st, pd, bias=True))
            self.output_layers.append(name)

    def forward(
            self,
            x: torch.tensor
    ):
        return [getattr(self, layer)(x) for layer in self.output_layers]


class _RSUNetMulti(nn.Module):
    """
    Full model for multiclass segmentation. Trained with 4 output channels:
        -   Ch0 = background
        -   Ch1 = soma
        -   Ch2 = axon
        -   Ch3 = dendrite
    """
    def __init__(
            self,
            d_in: int = 1,  # convention, N input dims / feature dims
            output_spec: OrderedDict = None,
            # output_spec: OrderedDict = OrderedDict(soma_label=1),  RSUNet
            depth: int = 4,
            io_size: tuple = IO_SIZE,
            io_stride: tuple = IO_STRIDE,
            bn: bool = BN
    ):
        super(_RSUNetMulti, self).__init__()
        output_spec = (
            OrderedDict(label=4) if output_spec is None else output_spec)

        assert depth < len(NFEATURES)
        self.depth = depth

        # Input feature embedding without batchnorm
        fs = NFEATURES[0]
        self.inputconv = _AConv(d_in, fs, io_size, st=io_stride)
        d_in = fs

        # modules in up/down paths added with setattr, obscured by U3D methods
        # Contracting pathway
        for d in range(depth):
            fs = NFEATURES[d]
            ks = SIZES[d]
            self.add_conv_mod(d, d_in, fs, ks, bn)
            self.add_max_pool(d+1)
            d_in = fs

        # Bridge
        fs = NFEATURES[depth]
        ks = SIZES[depth]
        self.add_conv_mod(depth, d_in, fs, ks, bn)
        d_in = fs

        # Expanding pathway
        for d in reversed(range(depth)):
            fs = NFEATURES[d]
            ks = SIZES[d]
            self.add_deconv_mod(d, d_in, fs, bn, ks)
            d_in = fs

        # Output feature embedding without batchnorm
        self.embedconv = _AConv(d_in, d_in, ks, st=(1, 1, 1))
        self.outputdeconv = _OutputModule(
            d_in, output_spec, ks=io_size, st=io_stride)

    def add_conv_mod(
            self,
            depth: int,
            d_in: int,
            d_out: int,
            ks: tuple,
            bn: bool
    ):
        setattr(self, f"convmod{depth}", _ConvMod(d_in, d_out, ks, bn=bn))

    def add_max_pool(
            self,
            depth: int,
            down: tuple = (2, 2, 2)
    ):
        setattr(self, f"maxpool{depth}", nn.MaxPool3d(down))

    def add_deconv_mod(
            self,
            depth: int,
            d_in: int,
            d_out: int,
            bn: bool,
            up: tuple = (2, 2, 2)
    ):
        setattr(self, f"deconv{depth}", _ConvTMod(d_in, d_out, up, bn=bn))

    def forward(
            self,
            x: torch.tensor
    ):
        # Input feature embedding without batchnorm
        # print(f"original: {x.shape}")
        x = self.inputconv(x)
        # print(f"input conv: {x.shape}")

        # Contracting pathway
        skip = []
        for d in range(self.depth):
            cd = getattr(self, f"convmod{d}")(x)
            x = getattr(self, f"maxpool{d + 1}")(cd)
            skip.append(cd)

            # print(f"convmod{d}: {cd.shape}")
            # print(f"maxpool{d + 1}: {x.shape}")

        # Bridge
        x = getattr(self, f"convmod{self.depth}")(x)
        # print(f"convmod{self.depth}: {x.shape}")

        # Expanding pathway
        for d in reversed(range(self.depth)):
            x = getattr(self, f"deconv{d}")(x, skip[d])
            # print(f"deconv{d}: {x.shape}")

        # Output feature embedding without batchnorm
        return self.outputdeconv(self.embedconv(x))[0]


class Monkey2DRSUNetMulti(_RSUNetMulti):
    def __init__(
            self
    ):
        super(Monkey2DRSUNetMulti, self).__init__()

    def forward(
            self,
            x: torch.tensor
    ):
        x = x.unsqueeze(2).expand(-1, -1, 32, -1, -1)
        x = super().forward(x)
        x = x[:, :, *tuple([s // 2 for s in x.size()[2:]])]
        x = torch.stack([x[:, 0], torch.logsumexp(x[:, 1:], dim=1)], dim=1)
        return x


class Monkey3DRSUNetMulti(_RSUNetMulti):
    def __init__(
            self
    ):
        super(Monkey3DRSUNetMulti, self).__init__()

    def forward(
            self,
            x: torch.tensor
    ):
        x = x if x.size(2) <= 32 else x.narrow(2, (x.size(2) - 32) // 2, 32)
        x = super().forward(x)
        x = x[:, :, *tuple([s // 2 for s in x.size()[2:]])]
        x = torch.concatenate(
            [x[:, :2], torch.logsumexp(x[:, 2:], dim=1, keepdim=True)], dim=1)
        return x
