#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:00:00 2025
@author: Ike
"""


import torch
import torch.nn as nn

from ..modules import make_block, make_dense_net


"""
Symmetric U-Net architecture designed to segment neuronal membranes from
volumetric electron microscopy imaging stacks. Adapted from Lee et al., 2017
"Superhuman Accuracy on the SNEMI3D Connectomics Challenge."
https://doi.org/10.48550/arXiv.1706.00120

This model incorporates design constraints based on the anisotropy of
neuroimaging data, particularly along the z axis. Thus, data are never
down- or up-sampled along this dimension, only along the y and x axes.
Likewise, all residual modules in the network begin with an anisotropic
convolution operation to calculate local features within z-planes and
across the y-x plane.

Definitions:
    Layer - a neural network operation with trainable parameters.
    Block - a layer paired with some combination of batch normalization,
        nonlinear activation, and dropout functions.
    Module - a sequence of consecutive layers and blocks.
    Depth - the number of layers between network input and output.
    Scale - the number of downsample = upsample operations in the network.
    width - the number of convolutional feature maps in a module.
"""
# arguments used for initial and final anisotropic convolutions
CONTEXT = {"kernel_size": (1, 5, 5), "stride": 1, "padding": (0, 2, 2)}
# arguments used for up-sampling and down-sampling operations
UP_DOWN = {"kernel_size": (1, 2, 2), "stride": (1, 2, 2)}
# Feature map density of convolutional layers
CHANNELS = (28, 36, 48, 64, 80)
C_MID = 18


"""
Linear classifier network is adapted from Simonyan & Zisserman, 2015 "Very Deep
Convolutional Networks for Large-scale Image Recognition."
https://doi.org/10.48550/arXiv.1409.1556
"""
# node distribution of linear classification layers
LINEARS = (1000,)  # (4096, 4096, 1000)


class ResidualModule(
    nn.Module
):
    """
    Module architecture:
    -   in  --> t_0 = prep(in)
    -   ........└─▶ t_1 = conv(t_0)
    -   ............├─▶ t_2 = conv(conv(t_1))
    -   out <-- ◀── post(t_1 + t_2) ◀───────┘

    Where:
    -   t_0: tensor [B, c_i, Z, Y, X]
    -   t_1 + t_2: tensor [B, c_o, Z, Y, X]

    NOTE: all ResidualModule instances take 3D spatial tensors as input, of
    shape [Batch, Channels, Z, Y, X]. For modules using 2D convolutions, this
    3D input is treated as a stack of 2D spatial tensors. During the 2D forward
    pass, each Z slice is fed through the module independently of all others.
    The output tensors are then stacked to yield a final tensor of the same
    shape as the original.

    Attributes:
        prep (nn.Module | None):
            Optional preparatory operation.
        conv1 (nn.Module):
            First convolution block.
        conv2 (nn.Module):
            Second and third convolution blocks.
        post (nn.Module | None):
            Optional post-processing operation.
    """
    def __init__(
            self,
            c_i: int,
            c_o: int,
            full: bool,
            act: type,
            drop: float,
            prep: nn.Module = None,
            post: nn.Module = None
    ):
        """
        Args:
            c_i (int):
                Number of channels in input tensor.
            c_o (int):
                Number of output channels from final convolution block.
            full (bool):
                If True, use 3D convolutions. If false, use 2D convolutions.
            act (type):
                Nonlinear activation function to use.
            drop (float):
                Dropout probability.
            prep (nn.Module, optional):
                Preparatory operation before first layer in module.
                Defaults to None, in which case module begins with first
                anisotropic convolution layer.
            post (nn.Module, optional):
                Post-processing operation after final layer in module.
                Defaults to None, in which case module ends with residual
                addition operation.
        """
        super(ResidualModule, self).__init__()
        self.full = full
        self.prep, self.post = prep, post

        conv, k, p, n = (nn.Conv3d, (1, 3, 3), (0, 1, 1), "3d") if full else (
            nn.Conv2d, 3, 1, "2d")
        self.conv1 = make_block(
            conv(c_i, c_o, kernel_size=k, stride=1, padding=p),
            c_o, norm=n, act=act, drop=drop)
        self.conv2 = nn.Sequential(*[
            make_block(
                conv(c_o, c_o, kernel_size=3, stride=1, padding=1),
                c_o, norm=n, act=act, drop=drop) for _ in range(2)])

    def forward(
            self,
            x: torch.tensor
    ):
        # optional preparatory module
        x = self.prep(x) if self.prep is not None else x

        # initial convolution
        x = self.conv1(x) if self.full else torch.stack(
            [self.conv1(x[:, :, z]) for z in range(x.shape[2])], dim=2)

        # final convolution block with residual connection
        x += self.conv2(x) if self.full else torch.stack(
            [self.conv2(x[:, :, z]) for z in range(x.shape[2])], dim=2)

        # optional post-processing module
        x = self.post(x) if self.post is not None else x
        return x


class SNEMI3DUnet(
    nn.Module
):
    """
    Module architecture:
    -   in  --> e_0 = encode_module(in)
    -   ........├─▶ e_1 = encode_module(e_0)
    -   ........│...├─▶ ***intermediate encoders***
    -   ........│...│...├─▶ e_n = encode_module(e_n-1)
    -   ........│...│...│...└─▶ d_n-1 = decode_module(e_n) ────┐
    -   ........│...│...d_n-2 = decode_module(d_n-1 + e_n-1) ◀─┘
    -   ........│...***intermediate decoders***
    -   out <-- decode_module(d_0 + e_0)) ◀───┘

    Where:
    -   n: number of downsampling modules in encoder, len(convs) - 1
    -   in: tensor [B, c_i, Z, Y, X]
    -   e_n: tensor [B, c_o, Z, Y//2^n, X//2^n]
    -   out: e_n or list(e_0, ..., e_n)

    Attributes:
        depth (int):
            Number of paired down/upsample modules in encoder
        encode_n (ResidualModule):
            nth encoding module in model.
        decode_n (ResidualModule):
            nth decoding module in model.
    """
    def __init__(
            self,
            c_i: int,
            c_o: int,
            c_m: int = C_MID,
            c_l: tuple[int, ...] = CHANNELS,
            act: type = nn.ELU,
            drop: float = None,
    ):
        """
        Args:
            c_i (int):
                Number of input channels in first layer of model.
            c_o (int):
                Number of output channels in final layer of model. Corresponds
                to number of object classes in input images.
            act (type, optional):
                Nonlinear activation function within each neural network layer.
                Defaults to nn.ELU.
            drop (float, optional):
                Dropout probability. If None, dropout is disabled.
                Defaults to 0.1.
            c_l (tuple[int, ...], optional)
                Output channel sizes of convolutional layers.
                Defaults to None, in which case sizes are (28, 36, 48, 64, 80).
        """
        super(SNEMI3DUnet, self).__init__()
        self.depth = len(c_l)

        # paired anisotropic input/output modules
        self.encode_0 = ResidualModule(
            c_m, c_l[0], full=False, act=act, drop=drop, prep=make_block(
                nn.Conv3d(c_i, c_m, **CONTEXT), c_m, norm="3d",
                act=act, drop=drop))
        self.decode_0 = ResidualModule(
            c_l[0], c_m, full=False, act=act, drop=drop, post=make_block(
                nn.Conv3d(c_m, c_o, **CONTEXT), c_o, norm="3d", act=act,
                drop=drop))

        # create paired encoder and decoder modules
        for i in range(1, len(c_l)):
            self._add_module("encode", i, c_l[i - 1], c_l[i], act, drop)
            self._add_module("decode", i, c_l[i], c_l[i - 1], act, drop)

    def _add_module(
            self,
            arm: str,
            depth: int,
            c_i: int,
            c_o: int,
            act: type,
            drop: float
    ):
        """
        Module architecture:
        -   encode: e_1 = _ResidualModule(maxpool(e_0))
        -   decode: e_0 = convtranspose(_ResidualModule(e_1))

        where:
        -   e_0 = tensor [B, c_o, Z, Y, X]
        -   e_1 = tensor [B, c_o, Z, Y//2, X//2]

        Args:
            arm (str):
                One of two possible values:
                -   "encode" for downsampling (maxpool) module
                -   "decode" for upsampling (convtranspose) module.
            depth (int):
                Number of downsample operations between raw input and output
                of encode module.
            c_i (int):
                Number of input channels to encoding module.
            c_o (int):
                Number of output channels from encoding module.
            act (type):
                See __init__.
            drop (float):
                See __init__.
        """
        kwargs = {
            "encode": {"prep": nn.MaxPool3d(**UP_DOWN)},
            "decode": {"post": nn.ConvTranspose3d(c_o, c_o, **UP_DOWN)}}[arm]
        setattr(self, f"{arm}_{depth}", ResidualModule(
            c_i, c_o, full=True, act=act, drop=drop, **kwargs))

    def forward(
            self,
            x: torch.tensor,
            decode: bool = True
    ):
        """
        Forward pass.

        Args:
            x (torch.tensor):
                Input tensor.
            decode (bool, optional):
                If True, run output of encoding network through symmetric
                decoder.
                Defaults to True.

        Returns:
            (torch.tensor):
                Output tensor following these operations:
                -   decode False: x --> encoder
                -   decode True: x --> encoder --> decoder
        """
        out = {0: self.encode_0(x)}
        for i in range(1, self.depth):
            out[i*int(decode)] = getattr(self, f"encode_{i}")(
                out[(i-1)*int(decode)])

        if decode:
            out[self.depth - 1] = getattr(
                self, f"decode_{self.depth - 1}")(out[self.depth - 1])
            for i in range(self.depth - 2, -1, -1):
                out[i] = getattr(self, f"decode_{i}")(out[i] + out[i+1])

        return out[0]


class SNEMI3DClassifier(
    nn.Module
):
    """
    Module architecture:
    -   in  --> t_0 = avgpool(SNEMI3DUnet_encoder(in)) ──┐
    -   out <-- ◀── dense_network(t_1) ◀─────────────────┘

    Where:
    -   in = tensor [B, c_i, Z, Y, X]
    -   t_0 = tensor [B, c_i, Z, 1, 1]
    -   out = tensor [B, c_i, Z, 1, 1]

    Attributes:
        encoder (SNEMI3DUnet):
            Encoding model.
        pooling (nn.Module):
            pooling module.
        percept (nn.Module):
            Dense linear classifier.
    """
    def __init__(
            self,
            c_i: int,
            v_i: int,
            c_o: int,
            c_l: tuple[int, ...] = CHANNELS,
            d_l: tuple[int, ...] = LINEARS,
            act_c: type = nn.ELU,
            act_l: type = nn.ReLU,
            drop_c: float = None,
            drop_l: float = None,
            **kwargs
    ):
        """
        Args:
            c_i (int):
                Number of input channels to model.
            v_i (int):
                Number of input z-planes to model.
            c_o (int):
                Number of output classes from model.
            c_l (tuple[int, ...], optional)
                Output channel sizes of convolutional layers.
                Defaults to None, in which case sizes are (28, 36, 48, 64, 80).
            d_l (tuple[int, ...], optional)
                Output sizes of dense linear layers.
                Defaults to None, in which case sizes are (4096, 4096, 1000).
            act_c (type, optional):
                Nonlinear activation function within each convolutional layer.
                Defaults to nn.ELU.
            act_l (type, optional):
                Nonlinear activation function within each linear layer.
                Defaults to nn.ReLU.
            drop_c (float, optional):
                Dropout probability within each convolutional layer.
                Defaults to None, in which case dropout is disabled.
            drop_l (float, optional):
                Dropout probability within each linear layer.
                Defaults to None, in which case dropout is disabled.
        """
        super(SNEMI3DClassifier, self).__init__()
        d_l = [c_l[-1] * v_i, *d_l]

        # create encoder, global pool, and dense linear modules
        self.encoder = SNEMI3DUnet(
            c_i=c_i, c_o=c_i, c_l=c_l, act=act_c, drop=drop_c)
        self.pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.percept = nn.Sequential(
            make_dense_net(d_l, norm=True, act=act_l, drop=drop_l),
            nn.Linear(d_l[-1], c_o))

    def forward(
            self,
            x: torch.tensor
    ):
        """
        Forward pass.

        Args:
            x (torch.tensor):
                Input tensor.

        Returns:
            (torch.tensor):
                Output tensor.
        """
        x = self.pooling(self.encoder(x, decode=False))
        return self.percept(torch.flatten(x, start_dim=1))
