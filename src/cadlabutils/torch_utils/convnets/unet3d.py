#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:00:00 2025
@author: Ike
"""


import torch
import torch.nn as nn

from ..modules import make_block, make_dense


"""
U-Net architecture designed for volumetric segmentation of sparsely annotated
volumetric images. Adapted from Çiçek et al., 2016
"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation."
https://doi.org/10.48550/arXiv.1606.06650
"""
# arguments used for up-sampling and down-sampling operations
UP_DOWN = {"kernel_size": 2, "stride": 2}
# Feature map density of convolutional layers
C_MID, CHANNELS, NODES = 32, (64, 128, 256, 512), (1000,)


class UNet3D(
    nn.Module
):
    """
    Module architecture:
    -   in  --> e_0 = encode_module(in)
    -   ........├─▶ e_1 = encode_module(e_0)
    -   ........│...├─▶ ***intermediate encoders***
    -   ........│...│...├─▶ e_n = encode_module(e_n-1)
    -   ........│...│...│...└─▶ d_n-1 = bottom_module(e_n) ───────┐
    -   ........│...│...d_n-2 = decode_module(cat(e_n-1, d_n-1) ◀─┘
    -   ........│...***intermediate decoders*** ─────┐
    -   out <-- conv(decode_module(cat(e_0, d_0))) ◀─┘

    *   n: number of downsampling modules in encoder
    *   in: tensor [B, c_i, Z, Y, X]
    *   e_n: tensor [B, c_o, Z//2^n, Y//2^n, X//2^n]
    *   out: e_n or list(e_0, ..., e_n)

    Attributes:
        depth (int):
            Number of paired down/upsample modules in encoder - 1
        encode_n (nn.Module):
            nth encoding module in model.
        decode_n (nn.Module):
            nth decoding module in model.
        bottom (nn.Module):
            Linker module between final encoding and first decoding module.
        final (nn.Module):
            Final convolution to specified number of output channels.
    """
    def __init__(
            self,
            c_i: int,
            c_o: int,
            c_m: int = C_MID,
            c_t: tuple[int] = CHANNELS,
            act: type = nn.ReLU,
            drop: float = None,
    ):
        """
        Args:
            c_i (int):
                Number of input channels in first layer of model.
            c_o (int):
                Number of output channels in final layer of model.
            c_m (int, optional):
                Intermediate channel count in first and final modules.
                Defaults to 18.
            c_t (tuple[int], optional)
                Output channel sizes of convolutional layers.
                Defaults to (64, 128, 256, 512).
            act (type, optional):
                Nonlinear activation function within each neural network layer.
                Defaults to nn.ELU.
            drop (float, optional):
                Dropout probability. If None, dropout is disabled.
                Defaults to 0.1.
        """
        super(UNet3D, self).__init__()
        self.depth, self.act, self.drop = len(c_t) - 1, act, drop

        # paired preparatory and post-processing convolution modules
        self._extend("encode_0", [c_i, c_m, c_t[0]])
        self._extend("decode_0", [sum(c_t[:2]), c_t[0], c_t[0]])

        # create paired encoder and decoder modules
        for i, c in enumerate(c_t[1:-1]):
            self._extend(f"encode_{i + 1}", [c_t[i], c_t[i], c], encode=True)
            self._extend(f"decode_{i + 1}", [c + c_t[i + 2], c, c], decode=True)

        # create symmetric bottom module and final 1x1x1 convolution
        self._extend(
            "bottom", [c_t[-2], c_t[-2], c_t[-1]], encode=True, decode=True)
        self.final = nn.Conv3d(c_t[0], c_o, kernel_size=1, stride=1)

    def _extend(
            self,
            name: str,
            c_l: list[int],
            encode: bool = False,
            decode: bool = False
    ):
        """
        Add module to model.

        Module architecture:
        -   encode True: e_1 = conv(maxpool(e_0))
        -   decode True: e_0 = convtranspose(conv(e_1))
        -   encode = decode: e_2 = conv(e_0)

        *   e_0 = tensor [B, c_l[0], Z, Y, X]
        *   e_1 = tensor [B, c_l[-1], Z//2, Y//2, X//2]
        *   e_2 = tensor [B, c_l[-1], Z, Y, X]

        Args:
            name (str):
                attribute name for created module.
            c_l (list[int]):
                Sequential channel counts of tensors within module.
            encode (bool, optional):
                If True, perform spatial downsampling with maxpooling.
                Defaults to False.
            decode (bool, optional):
                If True, perform spatial upsampling with convtranspose.
                Defaults to False.
        """
        lay = [nn.MaxPool3d(**UP_DOWN)] if encode else []
        lay += [
            make_block(
                nn.Conv3d(c_l[i], c, kernel_size=3, stride=1, padding=1),
                c, norm="3d", act=self.act, drop=self.drop)
            for i, c in enumerate(c_l[1:])]
        lay += [
            nn.ConvTranspose3d(c_l[-1], c_l[-1], **UP_DOWN)] if decode else []
        setattr(
            self, name, nn.Sequential(*lay) if len(lay) > 1 else lay[0])

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
        d = int(decode)

        # encoding modules
        out = {0: self.encode_0(x)}
        for i in range(1, self.depth):
            out[i*d] = getattr(self, f"encode_{i}")(out[(i-1)*d])

        # symmetric bottom module
        out[self.depth*d] = self.bottom(out[max(out.keys())])

        # decoding modules
        if decode:
            for i in range(self.depth - 1, -1, -1):
                out[i] = getattr(self, f"decode_{i}")(
                    torch.cat((out[i], out[i+1]), dim=1))

            # final 1x1x1 convolution
            out[0] = self.final(out[0])

        return out[0]


class UNet3DClassifier(
    UNet3D
):
    """
    Model architecture:
    -   in  --> t_0 = avgpool(UNet3D_encoder(in)) ──┐
    -   out <-- ◀── dense_network(t_1) ◀────────────┘

    *   in = tensor [B, c_i, Z, Y, X]
    *   t_0 = tensor [B, c_i, 1, 1, 1]
    *   out = tensor [B, c_o]

    Attributes:
        pool (nn.Module):
            pooling module.
        dense (nn.Module):
            Dense linear classifier.
    """
    def __init__(
            self,
            c_i: int,
            c_o: int,
            c_t: tuple[int] = CHANNELS,
            d_l: tuple[int] = NODES,
            act_c: type = nn.ReLU,
            act_l: type = nn.ReLU,
            drop_c: float = None,
            drop_l: float = None,
            **kwargs
    ):
        """
        Instantiate ResidualUNet model.

        Args:
            voxel (tuple[int]):
                Input size of model. Of shape (z, y, x).
            c_i (int):
                Number of input channels in first layer of model.
            c_o (int):
                Number of output classes in final layer of model.
            act_c (type, optional):
                Nonlinear activation function within each convolutional layer.
                Defaults to nn.ELU.
            act_l (type, optional):
                Nonlinear activation function within each linear layer.
                Defaults to nn.ReLU.
            c_t (tuple[int], optional)
                Output channel sizes of convolutional layers.
                Defaults to None, in which case sizes are (28, 36, 48, 64, 80).
            d_l (tuple[int], optional)
                Output sizes of dense linear layers.
                Defaults to None, in which case sizes are (4096, 4096, 1000).
            drop_c (float, optional):
                Dropout probability within each convolutional layer.
                Defaults to None, in which case dropout is disabled.
            drop_l (float, optional):
                Dropout probability within each linear layer.
                Defaults to None, in which case dropout is disabled.
        """
        super(UNet3DClassifier, self).__init__(
            c_i, c_i, c_t=c_t, act=act_c, drop=drop_c)

        # create pooling and dense linear layers
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dense = nn.Sequential(
            make_dense([c_t[-1], *d_l], norm=True, act=act_l, drop=drop_l),
            nn.Linear(d_l[-1], c_o))

    def forward(
            self,
            x: torch.tensor,
            **kwargs
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
        x = self.pool(super(UNet3DClassifier, self).forward(x, decode=False))
        return self.dense(torch.flatten(x, start_dim=1))
