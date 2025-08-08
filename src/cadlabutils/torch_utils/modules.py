#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:00:00 2025
@author: Ike
"""


import torch.nn as nn


NORM = {"1d": nn.BatchNorm1d, "2d": nn.BatchNorm2d, "3d": nn.BatchNorm3d}


def make_block(
        layer: nn.Module,
        c_out: int,
        norm: str = None,
        act: type = None,
        drop: float = None
):
    """
    Helper function to create a neural network block. In the full case, layer
    is organized such that input --> layer --> batch normalization -->
    activation --> dropout.

    Args:
        layer (nn.Module):
            Instantiated neural network layer to include in block.
        c_out (int):
            Number of output channels in layer.
        norm (str, optional):
            Type of batch normalization to perform. Can be "1d", "2d", or "3d".
            Defaults to None, in which case normalization is disabled.
        act (type, optional):
            Nonlinear activation function to use.
            Defaults to None, in which case activation function is disabled.
        drop (float, optional):
            Dropout probability.
             Defaults to None, in which case dropout is disabled.

    Returns:
        (nn.Module):
            Neural network block.
    """
    layers = [
        layer,
        NORM[norm](c_out) if norm is not None else None,
        act() if act is not None else None,
        nn.Dropout(drop) if drop is not None else None]
    return nn.Sequential(*[x for x in layers if x is not None])


def make_dense_net(
        c_all: list[int, ...],
        norm: bool,
        act: type = None,
        drop: float = None
):
    """
    Create a sequential neural network with dense layers.

    Args:
        c_all (list):
            Channels in each dense layer. Total network has len(channels) - 1
            layers, where layer[i] has channels[i] input and channels[i + 1]
            output channels.
        norm (bool):
            If True, perform batch normalization.
        act (type, optional):
            Passed to make_block function call.
            Defaults to None.
        drop (float, optional):
            Passed to make_block function call.
            Defaults to None.

    Returns:
        (nn.Module):
            Sequential neural network made of dense layers.
    """
    norm = "1d" if norm else None
    mlp = [
        make_block(nn.Linear(c_all[i], c), c, norm=norm, act=act, drop=drop)
        for i, c in enumerate(c_all[1:])]
    mlp = mlp[0] if len(mlp) == 1 else nn.Sequential(*mlp)
    return mlp
