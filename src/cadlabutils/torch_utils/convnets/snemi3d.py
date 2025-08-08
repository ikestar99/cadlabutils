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


class _ResidualModule(
    nn.Module
):
    """
    Module architecture:
    -   in  ──▶ t_0 = prep(in) ──▶ t_1 = conv0(t_0)
                                  ├─▶ t_2 = conv1(conv2(t_1))
    -   out ◀── post(t_3) ◀────── t_3 = t_1 + t_2 ──────────┘

    where:
    -   t_0 = torch.tensor of size [B, c_i, Z, Y, X]
    -   t_3 = torch.tensor of size [B, c_o, Z, Y, X]

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
        super(_ResidualModule, self).__init__()
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
        x = self.prep(x) if self.prep is not None else x
        return x


class _Encoder(
    nn.Module
):
    """
    Module architecture:
    -   in  ──▶ t_0 = conv(_ResidualModule(in))
                ├─▶ t_1 = maxpool(_ResidualModule(t_1))
                │   ├─▶ ...
                │   │   ├─▶ t_n = maxpool(_ResidualModule(t_n-1))
    -   out ◀── [t_0, t_1, ..., t_n-1, t_n] ────────────────────┘

    where:
    -   n = number of downsampling modules in encoder, len(convs) - 1
    -   in = torch.tensor of size [B, c_i, Z, Y, X]
    -   t_n = torch.tensor of size [B, c_o, Z, Y//2^n, X//2^n]

    Attributes:
        encoder (list[nn.Module, ...]):
            2D and 3D encoding modules.
    """
    def __init__(
            self,
            c_i: int,
            convs: tuple[int, ...],
            act: type,
            drop: float
    ):
        """
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
        super(_Encoder, self).__init__()
        self.encoder = list(range(len(convs)))

        self.encoder[0] = _ResidualModule(
            c_i, C_MID, full=False, act=act, drop=drop, prep=make_block(
                nn.Conv3d(C_MID, convs[0], **CONTEXT), C_MID, norm="3d",
                act=act, drop=drop))
        for i, c in enumerate(convs[1:]):
            self.encoder[i+1] = _ResidualModule(
                convs[i], c, full=True, act=act, drop=drop,
                prep=nn.MaxPool3d(**UP_DOWN))

    def forward(
            self,
            x: torch.tensor,
            as_list: bool = False
    ):
        """
        Forward pass through encoder.

        Args:
            x (torch.tensor):
                Input tensor. Of shape [Batch, c_i, Z, Y, X].
            as_list (bool, optional):
                If True, return intermediate values as tensors.
                Defaults to False, in which case only final output tensor
                is returned.

        Returns:
            (list | torch.tensor):
                Given that as_list arg is:
                -   True, value is list of intermediate tensors. List has
                    one output tensor per module, with the last index
                    corresponding to the final output of the encoder.
                -   False, value is final output tensor of encoder.
        """
        out_list = []
        for e in self.encoder:
            x = e(x)
            out_list += [x] if as_list else []

        return out_list if as_list else x


class _Decoder(
    nn.Module
):
    """
    Module architecture:
    -   in  ──▶ t_0 = conv(_ResidualModule(in))
                ├─▶ t_1 = maxpool(_ResidualModule(t_1))
                │   ├─▶ ...
                │   │   ├─▶ t_n = maxpool(_ResidualModule(t_n-1))
    -   out ◀── [t_0, t_1, ..., t_n-1, t_n] ────────────────────┘

    where:
    -   n = number of downsampling modules in encoder, len(convs) - 1
    -   in = torch.tensor of size [B, c_i, Z, Y, X]
    -   t_n = torch.tensor of size [B, c_o, Z, Y//2^n, X//2^n]

    Attributes:
        decoder (list[nn.Module, ...]):
            2D and 3D decoding modules.
    """
    def __init__(
            self,
            c_o: int,
            convs: tuple[int, ...],
            act: type,
            drop: float
    ):
        """
        Args:
            c_o (int):
                Number of output channels.
            convs (tuple[int, ...]):
                Output channel sizes of each convolutional module.
            act (type):
                Passed to make_block function call.
            drop (float):
                Passed to make_block function call.

        Returns:
            encoder (tuple[nn.Module, ...]):
                Modules included in encoder, ordered from shallow to deep.
        """
        super(_Decoder, self).__init__()
        self.decoder = list(range(len(convs)))

        self.decoder[0] = _ResidualModule(
            convs[0], C_MID, full=False, act=act, drop=drop, prep=make_block(
                nn.Conv3d(C_MID, c_o, **CONTEXT), C_MID, norm="3d", act=act,
                drop=drop))
        for i, c in enumerate(convs[:-1]):
            self.decoder[i] = _ResidualModule(
                convs[i+1], c, full=True, act=act, drop=drop,
                post=nn.ConvTranspose3d(c, c, **UP_DOWN))

    def forward(
            self,
            x_list: list[torch.tensor, ...]
    ):
        """
        Forward pass through encoder.

        Args:
            x_list (list[torch.tensor, ...]):
                Tensors output by symmetric encoder.

        Returns:
            (torch.tensor):
                Final output tensor of decoder.
        """
        x = x_list[-1]
        for i in range(-1, -len(x_list) - 1, -1):
            x = self.decoder[i](x if i == -1 else x + x_list[i])

        return x


class SNEMI3D_C(
    nn.Module
):
    """
    Module architecture
    ---------------------------------------------------------------------------
    in
    └─▶ t0 = _2d_encode(in)
        └─▶ t1 = conv(conv(t0)

    t0 + t1 ◀────────────────┘
    ---------------------------------------------------------------------------
    in: [B, c_i, Z, Y, X], out: [B, c_o]

    NOTE: SNEMI3D architecture modified for voxel-wise classification

    """
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

        # create encoder, global pool, and dense linear modules
        self.encoder = _Encoder(c_i, convs, act=act_c, drop=drop_c)
        self.pooling = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.linears = nn.Sequential(
            make_dense_net(dense, norm=True, act=act_l, drop=drop_l),
            nn.Linear(dense[-1], c_o))

    def forward(
            self,
            x: torch.tensor
    ):
        x = self.pooling(self.encoder(x, as_list=False))
        return self.linears(torch.flatten(x, start_dim=1))


class SNEMI3D_U(
    nn.Module
):
    """SNEMI3D architecture modified for semantic segmentation."""
    def __init__(
            self,
            c_i: int,
            c_o: int,
            act: type = nn.ELU,
            drop: float = None,
            convs: tuple[int, ...] = None,
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
            convs (tuple[int, ...], optional)
                Output channel sizes of convolutional layers.
                Defaults to None, in which case sizes are (28, 36, 48, 64, 80).
        """
        super(SNEMI3D_U, self).__init__()
        convs = CHANNELS if convs is None else convs

        # create encoder and decoder modules
        self.encoder = _Encoder(c_i, convs, act=act, drop=drop)
        self.decoder = _Decoder(c_o, convs, act=act, drop=drop)

    def forward(
            self,
            x: torch.tensor
    ):
        return self.decoder(self.encoder(x, as_list=True))


def __main__():
    voxel = (16, 32, 32)
    model_c = SNEMI3D_C(c_in=1, v_i=32, c_out=3)
    model_3 = SNEMI3D_U(c_in=1, c_out=2)

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
