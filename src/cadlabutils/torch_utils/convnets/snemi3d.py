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
C_MIDDLE = 18


"""
Linear classifier network is adapted from Simonyan & Zisserman, 2015 "Very Deep
Convolutional Networks for Large-scale Image Recognition."
https://doi.org/10.48550/arXiv.1409.1556
"""
# node distribution of linear classification layers
LINEARS = (1000,)  # (4096, 4096, 1000)


class _ResidualModule(
    nn.Module
):
    """
    Simple residual module built on 2 or 3D convolution blocks. Includes a
    residual skip (addition) connections between outputs of first and last
    convolution blocks.

    NOTE: all ResidualModule instances take 3D spatial tensors as input, of
    shape [Batch, Channels, Z, Y, X]. For modules using 2D convolutions, this
    3D input is treated as a stack of 2D spatial tensors. During the 2D forward
    pass, each Z slice is fed through the module independently of all others.
    The output tensors are then stacked to yield a final tensor of the same
    shape as the original.

    Attributes:
        conv1 (nn.Module):
            First convolution block.
        conv2 (nn.Module):
            Second and third convolution blocks.
    """
    def __init__(
            self,
            c_in: int,
            c_out: int,
            full: bool,
            act: type,
            drop: float
    ):
        """
        Args:
            c_in (int):
                Number of channels in input tensor.
            c_out (int):
                Number of output channels from final convolution block.
            full (bool):
                If True, use 3D convolutions. If false, use 2D convolutions.
            act (type):
                Nonlinear activation function to use.
            drop (float):
                Dropout probability.
        """
        super(_ResidualModule, self).__init__()
        self.full = full

        conv, k, p, n = (nn.Conv3d, (1, 3, 3), (0, 1, 1), "3d") if full else (
            nn.Conv2d, 3, 1, "2d")
        self.conv1 = make_block(
            conv(c_in, c_out, kernel_size=k, stride=1, padding=p),
            c_out, norm=n, act=act, drop=drop)
        self.conv2 = nn.Sequential(*[
            make_block(
                conv(c_out, c_out, kernel_size=3, stride=1, padding=1),
                c_out, norm=n, act=act, drop=drop) for _ in range(2)])

    def forward(
            self,
            x: torch.tensor  # <B, C, Z, Y, X>
    ):
        if self.full:
            x = self.conv1(x)
            o = self.conv2(x)
        else:
            x = torch.stack(
                [self.conv1(x[:, :, z]) for z in range(x.shape[2])], dim=2)
            o = torch.stack(
                [self.conv2(x[:, :, z]) for z in range(x.shape[2])], dim=2)

        return x + o


def _2d_module(
        c_i: int,
        c_m: int,
        c_o: int,
        act: type,
        drop: float,
        encode: bool
):
    """
    Create 2D anisotropic module for use in encoder or decoder.

    Args:
        c_i (int):
            Number of input channels.
        c_m (int):
            Number of intermediate channels.
        c_o (int):
            Number of output channels.
        act (type):
            Passed to make_block function call.
        drop (float):
            Passed to make_block function call.
        encode (bool):
            If True, preparatory convolution comes before residual module. If
            False, order is reversed.

    Returns:
        module (nn.Module):
            Module of specified type.
    """
    c0, c1 = ([c_i, c_m], [c_m, c_o]) if encode else ([c_m, c_o], [c_i, c_m])
    module = [
        make_block(
            nn.Conv3d(*c0, **CONTEXT), c0[1], norm="3d", act=act, drop=drop),
        _ResidualModule(*c1, full=False, act=act, drop=drop)]
    return nn.Sequential(*(module if encode else module[::-1]))


def _3d_module(
        c_i: int,
        c_o: int,
        act: type,
        drop: float,
        encode: bool
):
    """
    Create 3D anisotropic module for use in encoder or decoder.

    Args:
        c_i (int):
            Number of input channels.
        c_o (int):
            Number of output channels.
        act (type):
            Passed to make_block function call.
        drop (float):
            Passed to make_block function call.
        encode (bool):
            If True, return encoder module with spatial downsampling. If False,
            return decoder module with spatial upsampling.

    Returns:
        module (nn.Module):
            Module of specified type.
    """
    module = [
        nn.MaxPool3d(**UP_DOWN) if encode else None,
        _ResidualModule(c_i, c_o, full=True, act=act, drop=drop),
        nn.ConvTranspose3d(c_o, c_o, **UP_DOWN) if not encode else None]
    return nn.Sequential(*[x for x in module if x is not None])


def _encoder(
        c_i: int,
        convs: tuple[int, ...],
        act: type,
        drop: float
):
    """
    Create encoder with stacked 2D and 3D encoding modules.

    Args:
        c_i (int):
            Number of input channels.
        convs (tuple[int, ...]):
            Output channel sizes of convolutional layers.
        act (type):
            Passed to make_block function call.
        drop (float):
            Passed to make_block function call.

    Returns:
        encoder (tuple[nn.Module, ...]):
            Modules included in encoder, ordered from shallow to deep.
    """
    encoder = [
        _2d_module(c_i, C_MIDDLE, convs[0], act=act, drop=drop, encode=True)]
    encoder += [
        _3d_module(convs[i], c, act=act, drop=drop, encode=True)
        for i, c in enumerate(convs[1:])]
    return tuple(encoder)


class SNEMI3D_C(
    nn.Module
):
    """SNEMI3D architecture modified for voxel classification."""
    def __init__(
            self,
            c_i: int,
            v_i: int,
            c_o: int,
            act_c: type = nn.ELU,
            act_l: type = nn.ReLU,
            convs: tuple[int, ...] = None,
            dense: tuple[int, ...] = None,
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
            act_c (type, optional):
                Nonlinear activation function within each convolutional layer.
                Defaults to nn.ELU.
            act_l (type, optional):
                Nonlinear activation function within each linear layer.
                Defaults to nn.ReLU.
            convs (tuple[int, ...], optional)
                Output channel sizes of convolutional layers.
                Defaults to None, in which case sizes are (28, 36, 48, 64, 80).
            dense (tuple[int, ...], optional)
                Output sizes of dense linear layers.
                Defaults to None, in which case sizes are (4096, 4096, 1000).
            drop_c (float, optional):
                Dropout probability within each convolutional layer.
                Defaults to None, in which case dropout is disabled.
            drop_l (float, optional):
                Dropout probability within each linear layer.
                Defaults to None, in which case dropout is disabled.
        """
        super(SNEMI3D_C, self).__init__()
        convs = CHANNELS if convs is None else convs
        dense = [convs[-1] * v_i, *(LINEARS if dense is None else dense)]

        # create encoder
        self.e = nn.Sequential(
            *_encoder(c_i, convs, act=act_c, drop=drop_c))

        # create adaptive pool layer across final 2 dimensions
        self.p = nn.AdaptiveAvgPool3d((None, 1, 1))

        # create multi-layer perceptron classifier
        self.d = nn.Sequential(
            make_dense_net(dense, norm=True, act=act_l, drop=drop_l),
            nn.Linear(dense[-1], c_o))

    def forward(
            self,
            x: torch.tensor
    ):
        return self.d(torch.flatten(self.p(self.e(x)), start_dim=1))


class UNetSNEMI3D(
    nn.Module
):
    """SNEMI3D architecture modified for semantic segmentation."""
    def __init__(
            self,
            c_in: int,
            c_out: int,
            act: type = nn.ELU,
            drop: float = None,
            convs: tuple[int, ...] = None,
    ):
        """
        Args:
            c_in (int):
                Number of input channels in first layer of model.
            c_out (int):
                Number of output channels in final layer of model. Corresponds
                to number of object classes in input images.
            act (type, optional):
                Nonlinear activation function within each neural network layer.
                Defaults to nn.ELU.
            drop (float, optional):
                Dropout probability. If None, dropout is disabled.
                Defaults to 0.1.
            convs (tuple[int, ...], optional)
                Output channel sizes of convolutional layers.
                Defaults to None, in which case sizes are (28, 36, 48, 64, 80).
        """
        super(UNetSNEMI3D, self).__init__()
        convs = CHANNELS if convs is None else convs
        self.depth = len(convs) - 1

        # preparatory anisotropic convolution module
        self.first = nn.Sequential(
            make_block(
                nn.Conv3d(c_in, convs[0], **CONTEXT),
                convs[0], norm="3d", act=act, drop=drop),
            _ResidualModule(
                convs[0], convs[0], full=False, act=act, drop=drop))

        # create down sampling modules
        for i, w in enumerate(convs[1:]):
            setattr(self, f"encode_{i}", nn.Sequential(
                nn.MaxPool3d(**UP_DOWN),
                _ResidualModule(convs[i], w, full=True, act=act, drop=drop)))

        # create up sampling modules
        for i, w in enumerate(convs[-2::-1]):
            setattr(self, f"decode_{i}", nn.Sequential(
                _ResidualModule(
                    convs[-(i + 1)], w, full=True, act=act, drop=drop),
                nn.ConvTranspose3d(w, w, **UP_DOWN)))

        # final anisotropic convolution module
        self.final = nn.Sequential(
            _ResidualModule(
                convs[0], convs[0], full=False, act=act, drop=drop),
            make_block(
                nn.Conv3d(convs[0], c_out, **CONTEXT),
                c_out, norm="3d", act=act, drop=drop))

    def forward(
            self,
            x: torch.tensor
    ):
        # encoder arm, save intermediate outputs
        skips = [self.first(x)]
        for i in range(self.depth):
            skips += [getattr(self, f"encode_{i}")(skips[-1])]

        # decoder arm, add intermediate outputs
        for i, s in enumerate(skips[:0:-1]):
            x = getattr(self, f"decode_{i}")(s if i == 0 else x + s)

        return self.final(x + skips[0])


def __main__():
    voxel = (16, 32, 32)
    model_c = SNEMI3D_C(c_in=1, c_out=3)
    model_3 = UNetSNEMI3D(c_in=1, c_out=2)

    c_params = sum(p.numel() for p in model_c.parameters() if p.requires_grad)
    u_params = sum(p.numel() for p in model_3.parameters() if p.requires_grad)
    print(f"classifier: {c_params} params\nunet: {u_params} params")

    model_c.eval()
    model_3.eval()
    with torch.no_grad():
        import numpy as np
        temp = torch.from_numpy(np.random.rand(64, 1, *voxel)).float()
        out_c = model_c(temp)
        out_3 = model_3(temp)

        print(f"classifier output: {out_c.shape}\nunet output: {out_3.shape}")


if __name__ == "__main__":
    __main__()
