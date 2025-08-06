#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:00:00 2025
@author: Ike
"""


import torch
import torch.nn as nn

from ..modules import make_block


"""
U-Net architecture designed for volumetric segmentation of sparsely annotated
volumetric images. Adapted from Çiçek et al., 2016
"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation."
https://doi.org/10.48550/arXiv.1606.06650
"""
# Feature map density of convolutional layers
CHANNELS = (64, 128, 256, 512)
# node distribution of linear classification layers
LINEARS = (1000,)  # (4096, 4096, 1000)


class ConvModule(
    nn.Module
):
    """
    Simple module built on sequential 3D convolution blocks.

    Attributes:
        conv (nn.Module):
            Sequential convolution block.
    """
    def __init__(
            self,
            c_list: list[int],
            act: type,
            drop: float
    ):
        """
        Args:
            c_list (list[int]):
                Channels for each convolutional layer. Has structure
                [C_in, C_out_1, ..., C_out_n].
            act (type):
                Nonlinear activation function to use.
            drop (float):
                Dropout probability.
        """
        super(ConvModule, self).__init__()

        self.conv = nn.Sequential(*[
            make_block(
                nn.Conv3d(
                    c_list[i], c_out, kernel_size=3, stride=1, padding=1),
                c_out, norm="3d", act=act, drop=drop)
            for i, c_out in enumerate(c_list[1:])])

    def forward(
            self,
            x: torch.tensor,  # <B, C, Z, Y, X>
    ):
        return self.conv(x)


class UNet3DClassifier(
    nn.Module
):
    """3D U-Net architecture modified for voxel classification."""
    def __init__(
            self,
            c_in: int,
            c_out: int,
            act_c: type = nn.ReLU,
            act_l: type = nn.ReLU,
            convs: tuple[int, ...] = None,
            dense: tuple[int, ...] = None,
            drop_c: float = None,
            drop_l: float = None,
            **kwargs
    ):
        """
        Instantiate ResidualUNet model.

        Args:
            voxel (tuple[int, ...]):
                Input size of model. Of shape (z, y, x).
            c_in (int):
                Number of input channels in first layer of model.
            c_out (int):
                Number of output classes in final layer of model.
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
        super(UNet3DClassifier, self).__init__()
        convs = CHANNELS if convs is None else convs
        self.dense = LINEARS if dense is None else dense
        self.depth = len(convs) - 1

        # preparatory convolution module
        self.first = ConvModule(
            [c_in, convs[0] // 2, convs[0]], act=act_c, drop=drop_c)

        # create down sampling modules
        for i, w in enumerate(convs[1:]):
            setattr(self, f"encode_{i}", nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                ConvModule(
                    [convs[i], convs[i], w], act=act_c, drop=drop_c)))

        # create pool layer across final 2 dimensions
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        for i, w in enumerate(self.dense):
            c_i = convs[-1]  if i == 0 else self.dense[i - 1]
            setattr(self, f"linear_{i}", make_block(
                nn.Linear(c_i, w), w, norm="1d", act=act_l, drop=drop_l))

        self.final = nn.Linear(self.dense[-1], c_out)

    def forward(
            self,
            x: torch.tensor
    ):
        # encoder arm
        x = self.first(x)
        for i in range(self.depth):
            x = getattr(self, f"encode_{i}")(x)

        # dense linear layers
        x = torch.flatten(self.pool(x), start_dim=1)
        for i, _ in enumerate(self.dense):
            x = getattr(self, f"linear_{i}")(x)

        return self.final(x)


class Unet3D(
    nn.Module
):
    """3D U-Net architecture modified for semantic segmentation."""
    def __init__(
            self,
            c_in: int,
            c_out: int,
            act: type = nn.ReLU,
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
                Defaults to None, in which case sizes are (64, 128, 256, 512).
        """
        super(Unet3D, self).__init__()
        convs = CHANNELS if convs is None else convs
        self.depth = len(convs) - 1

        # preparatory convolution module
        self.first = ConvModule(
            [c_in, convs[0] // 2, convs[0]], act=act, drop=drop)

        # create down sampling modules
        for i, w in enumerate(convs[1:]):
            setattr(self, f"encode_{i}", nn.Sequential(
                nn.MaxPool3d(kernel_size=2, stride=2),
                ConvModule([convs[i], convs[i], w], act=act, drop=drop)))

        # create up sampling modules
        self.bottom = nn.ConvTranspose3d(w, w, kernel_size=2, stride=2)
        for i, w in enumerate(convs[-2:0:-1]):
            setattr(self, f"decode_{i}", nn.Sequential(
                ConvModule(
                    [convs[-(i + 1)] + w, w, w], act=act, drop=drop),
                nn.ConvTranspose3d(w, w, kernel_size=2, stride=2)))

        # final convolution layer
        self.final = nn.Sequential(
            ConvModule(
                [sum(convs[:2]), convs[0], convs[0]], act=act, drop=drop),
            nn.Conv3d(convs[0], c_out, kernel_size=1, stride=1, padding=0))

    def forward(
            self,
            x: torch.tensor
    ):
        # encoder arm, save intermediate outputs
        skips = [self.first(x)]
        for i in range(self.depth):
            skips += [getattr(self, f"encode_{i}")(skips[-1])]

        # decoder arm, concat intermediate outputs
        x = self.bottom(skips[-1])
        for i, s in enumerate(skips[-2:0:-1]):
            x = getattr(self, f"decode_{i}")(torch.cat((s, x), dim=1))

        return self.final(torch.cat((skips[0], x), dim=1))


def __main__():
    voxel = (16, 32, 32)
    model_c = UNet3DClassifier(voxel=voxel, c_in=1, c_out=3)
    model_3 = Unet3D(c_in=1, c_out=2)

    c_params = sum(p.numel() for p in model_c.parameters() if p.requires_grad)
    u_params = sum(p.numel() for p in model_3.parameters() if p.requires_grad)
    print(f"classifier: {c_params} params\nunet: {u_params} params")

    model_c.eval()
    model_3.eval()
    with torch.no_grad():
        import numpy as np
        temp = torch.from_numpy(np.random.rand(1, 1, *voxel)).float()
        out_c = model_c(temp)
        out_3 = model_3(temp)

        print(f"classifier output: {out_c.shape}\nunet output: {out_3.shape}")


if __name__ == "__main__":
    __main__()
