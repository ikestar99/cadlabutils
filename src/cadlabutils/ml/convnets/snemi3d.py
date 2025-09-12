#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:00:00 2025
@author: Ike
"""


import torch
import torch.nn as nn

from .. import make_block, make_dense


class ResidualModule(
    nn.Module
):
    """Simple module with residual skip connection.

    Attributes
    ----------
    prep : nn.Module | None
        Optional preparatory operation.
    conv1 : nn.Module
        First convolution block.
    conv2 : nn.Module
        Second and third convolution blocks.
    post : nn.Module | None
        Optional post-processing operation.

    Parameters
    ----------
    c_i : int
        Number of channels in input tensor.
    c_o : int
        Number of output channels from final convolution block.
    full : bool
        If True, use 3D convolutions. If false, use 2D convolutions.
    act : type
        Nonlinear activation function to use.
    drop : float
        Dropout probability.
    prep : nn.Module, optional
        Preparatory operation before first layer in module.
        Defaults to None.
    post : nn.Module, optional
        Post-processing operation after final layer in module.
        Defaults to None.

    Notes
    -----
    in  --> t_0 = prep(in)
    ........└─▶ t_1 = conv(t_0)
    ............├─▶ t_2 = conv(conv(t_1))
    out <-- ◀── post(t_1 + t_2) ◀───────┘

    *   t_0: tensor [B, c_i, Z, Y, X]
    *   t_1 + t_2: tensor [B, c_o, Z, Y, X]

    References
    ----------
    [1] Lee, K., Zung, J., Li, P., Jain, V. & Seung, H. S. Superhuman Accuracy
        on the SNEMI3D Connectomics Challenge. Preprint at
        https://doi.org/10.48550/arXiv.1706.00120 (2017).
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
        """Forward pass.

        Parameters
        ----------
        x : torch.tensor
            Shape is (Batch, `c_i`, Z, Y, X).

        Returns
        -------
        x : torch.tensor
            Shape is (Batch, `c_o`, Z, Y, X).

        Notes
        -----
        For modules using 2D convolutions, each Z slice is fed through the
        module independently of all others. The output tensors are then stacked
        back together along the z axis.
        """
        # optional preparatory module
        x = self.prep(x) if self.prep is not None else x

        # initial convolution
        x = self.conv1(x) if self.full else torch.stack(
            [self.conv1(x[:, :, z]) for z in range(x.shape[2])], dim=2)

        # final convolution block with residual connection
        x = x + self.conv2(x) if self.full else torch.stack(
            [self.conv2(x[:, :, z]) for z in range(x.shape[2])], dim=2)

        # optional post-processing module
        x = self.post(x) if self.post is not None else x
        return x


class SNEMI3D(
    nn.Module
):
    """SNEMI3D-based U-Net architecture.

    Class Attributes
    ----------------
    CONTEXT : dict[str, int | tuple[int, ...]]
        Arguments used for initial and final anisotropic convolutions
    UP_DOWN : dict[str, int]
        Arguments used for up-sampling and down-sampling operations.

    Attributes
    ----------
    depth : int
        Number of paired down/upsample modules in encoder
    encode_n : ResidualModule
        nth encoding module in model.
    decode_n : ResidualModule
        nth decoding module in model.

    Parameters
    ----------
    c_i : int
        Number of input channels in first layer of model.
    c_o : int
        Number of output channels in final layer of model.
    c_m : int, optional
        Intermediate channel count in first and final modules.
        Defaults to 18.
    c_t : tuple[int, ...], optional)
        Output channel sizes of convolutional layers.
        Defaults to (28, 36, 48, 64, 80).
    act : type, optional
        Nonlinear activation function within each neural network layer.
        Defaults to nn.ELU.
    drop : float, optional
        Dropout probability. If None, dropout is disabled.
        Defaults to 0.1.

    Notes
    -----
    in  --> e_0 = encode_module(in)
    ........├─▶ e_1 = encode_module(e_0)
    ........│...├─▶ ***intermediate encoders***
    ........│...│...├─▶ e_n = encode_module(e_n-1)
    ........│...│...│...└─▶ d_n-1 = decode_module(e_n) ────┐
    ........│...│...d_n-2 = decode_module(d_n-1 + e_n-1) ◀─┘
    ........│...***intermediate decoders***
    out <-- decode_module(d_0 + e_0)) ◀───┘

    *   n: number of downsampling modules in encoder
    *   in: tensor [B, c_i, Z, Y, X]
    *   e_n: tensor [B, c_o, Z, Y//2^n, X//2^n]
    *   out: e_n or list(e_0, ..., e_n)

    References
    ----------
    [1] Lee, K., Zung, J., Li, P., Jain, V. & Seung, H. S. Superhuman Accuracy
        on the SNEMI3D Connectomics Challenge. Preprint at
        https://doi.org/10.48550/arXiv.1706.00120 (2017).
    """
    CONTEXT = {"kernel_size": (1, 5, 5), "stride": 1, "padding": (0, 2, 2)}
    UP_DOWN = {"kernel_size": (1, 2, 2), "stride": (1, 2, 2)}

    def __init__(
            self,
            c_i: int,
            c_o: int,
            c_m: int = 18,
            c_t: tuple[int, ...] = (28, 36, 48, 64, 80),
            act: type = nn.ELU,
            drop: float = None,
            **kwargs
    ):
        super(SNEMI3D, self).__init__()
        self.depth = len(c_t)
        self.act = act
        self.drop = drop
        self.c_f = c_t[-1]

        # paired anisotropic input/output modules
        self.encode_0 = ResidualModule(
            c_m, c_t[0], full=False, act=act, drop=drop, prep=make_block(
                nn.Conv3d(c_i, c_m, **self.CONTEXT), c_m, norm="3d",
                act=self.act, drop=self.drop))
        self.decode_0 = ResidualModule(
            c_t[0], c_m, full=False, act=act, drop=drop, post=make_block(
                nn.Conv3d(c_m, c_o, **self.CONTEXT), c_o, norm="3d",
                act=self.act, drop=self.drop))

        # create paired encoder and decoder modules
        for i in range(1, len(c_t)):
            self._extend("encode", i, c_t[i - 1], c_t[i])
            self._extend("decode", i, c_t[i], c_t[i - 1])

    def _extend(
            self,
            arm: str,
            depth: int,
            c_i: int,
            c_o: int
    ):
        """Add module to model.

        Parameters
        ----------
        arm : str
            One of two possible values:
            -   "encode" for downsampling (maxpool) module
            -   "decode" for upsampling (convtranspose) module.
        depth : int
            Number of downsample operations between raw input and output
            of encode module.
        c_i : int
            Number of input channels to encoding module.
        c_o : int
            Number of output channels from encoding module.

        Notes
        -----
        encode: e_1 = _ResidualModule(maxpool(e_0))
        decode: e_0 = convtranspose(_ResidualModule(e_1))

        *   e_0 = tensor [B, c_o, Z, Y, X]
        *   e_1 = tensor [B, c_o, Z, Y//2, X//2]
        """
        kwargs = {
            "encode": {"prep": nn.MaxPool3d(**self.UP_DOWN)},
            "decode": {
                "post": nn.ConvTranspose3d(c_o, c_o, **self.UP_DOWN)}}[arm]
        setattr(self, f"{arm}_{depth}", ResidualModule(
            c_i, c_o, full=True, act=self.act, drop=self.drop, **kwargs))

    def forward(
            self,
            x: torch.tensor,
            decode: bool = True
    ):
        """Forward pass.

        Parameters
        ----------
        x : torch.tensor
            Input tensor.
        decode : bool, optional
            If True, run output of encoding network through symmetric decoder.
            Defaults to True.

        Returns
        -------
        torch.tensor
            Output tensor following these operations:
            -   decode False: x --> encoder
            -   decode True: x --> encoder --> decoder
        """
        # encoding modules
        out = {0: self.encode_0(x)}
        for i in range(1, self.depth):
            out[i*int(decode)] = getattr(self, f"encode_{i}")(
                out[(i-1)*int(decode)])

        # decoding modules
        if decode:
            out[self.depth - 1] = getattr(
                self, f"decode_{self.depth - 1}")(out[self.depth - 1])
            for i in range(self.depth - 2, -1, -1):
                out[i] = getattr(self, f"decode_{i}")(out[i] + out[i+1])

        return out[0]


class SNEMI3DClassifier(
    SNEMI3D
):
    """Extended SNEMI3D architecture for classification.

    Attributes
    ----------
    pool (nn.Module):
        pooling module.
    dense (nn.Module):
        Dense linear classifier.

    Parameters
    ----------
    c_i : int
        Number of input channels to model.
    v_i : int
        Number of input z-planes to model.
    c_o : int
        Number of output bases from model.
    c_t : tuple[int, ...], optional
        Output channel sizes of convolutional layers.
        See `SNEMI3D` `c_t` for default value.
    d_l : tuple[int, ...], optional
        Output sizes of dense linear layers.
        Defaults to None, in which case sizes are (4096, 4096, 1000).
    act_c : type, optional
        Nonlinear activation function within each convolutional layer.
        See `SNEMI3D` `act` for default value.
    act_l : type, optional
        Nonlinear activation function within each linear layer.
        Defaults to nn.ReLU.
    drop_c : float, optional
        Dropout probability within each convolutional layer.
        Defaults to None, in which case dropout is disabled.
    drop_l : float, optional
        Dropout probability within each linear layer.
        Defaults to None, in which case dropout is disabled.

    Notes
    -----
    in  --> t_0 = avgpool(SNEMI3DUnet_encoder(in)) ──┐
    out <-- ◀── dense_network(t_1) ◀─────────────────┘

    *   in = tensor [B, c_i, Z, Y, X]
    *   t_0 = tensor [B, c_i, Z, 1, 1]
    *   out = tensor [B, c_o]

    References
    ----------
    [1] Lee, K., Zung, J., Li, P., Jain, V. & Seung, H. S. Superhuman Accuracy
        on the SNEMI3D Connectomics Challenge. Preprint at
        https://doi.org/10.48550/arXiv.1706.00120 (2017).
    """
    def __init__(
            self,
            c_i: int,
            v_i: int,
            c_o: int,
            c_t: tuple[int, ...] = None,
            d_l: tuple[int, ...] = (1000,),
            act_c: type = None,
            act_l: type = nn.ReLU,
            drop_c: float = None,
            drop_l: float = None,
            **kwargs
    ):
        to_super = {c_t: c_t, "act": act_c}
        super(SNEMI3DClassifier, self).__init__(
            c_i, c_i, drop=drop_c,
            **{k: v for k, v in to_super.items() if v is not None})

        # add global pooling and dense linear modules
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.dense = nn.Sequential(
            make_dense([self.c_f*v_i, *d_l], norm=True, act=act_l, drop=drop_l),
            nn.Linear(d_l[-1], c_o))

    def forward(
            self,
            x: torch.tensor,
            **kwargs
    ):
        """Forward pass.

        Parameters
        ----------
        x : torch.tensor
            Input tensor.

        Returns
        -------
        torch.tensor
            Output tensor.
        """
        x = self.pool(super(SNEMI3DClassifier, self).forward(x, decode=False))
        return self.dense(torch.flatten(x, start_dim=1))
