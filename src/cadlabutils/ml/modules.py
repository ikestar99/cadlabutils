#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:00:00 2025
@author: Ike
"""


# 2. Third-party library imports
import torch.nn as nn


NORM = {"1d": nn.BatchNorm1d, "2d": nn.BatchNorm2d, "3d": nn.BatchNorm3d}


def make_block(
        layer: nn.Module,
        c_out: int,
        norm: str = None,
        act: type = None,
        drop: float = None
):
    """Helper function to create a neural network block.

    Parameters
    ----------
    layer : nn.Module
        Instantiated neural network layer to include in block.
    c_out : int
        Number of output channels in layer.
    norm : str, optional
        Type of batch normalization to perform. Can be "1d", "2d", or "3d".
        Defaults to None, in which case normalization is disabled.
    act : type, optional
        Nonlinear activation function to use.
        Defaults to None, in which case activation function is disabled.
    drop : float, optional
        Dropout probability.
        Defaults to None, in which case dropout is disabled.

    Returns
    -------
    nn.Module
        Neural network block.

    Notes
    -----
    In the full case, layer is organized such that input --> layer --> batch
    normalization --> activation --> dropout.

    Examples
    --------
    Block with dense layer and normalization.
    >>> test_c_out = 10
    >>> test_layer = nn.Linear(100, test_c_out)
    >>> make_block(test_layer, c_out=test_c_out, norm="1d")
    ... # doctest: +ELLIPSIS
    Sequential(
      (0): Linear(in_features=100, out_features=10, bias=True)
      (1): BatchNorm1d(10, ...)
    )

    Block with dense layer, normalization, activation, and dropout.
    >>> make_block(
    ...     test_layer, c_out=test_c_out, norm="1d", act=nn.ReLU, drop=0.5)
    ... # doctest: +ELLIPSIS
    Sequential(
      (0): Linear(in_features=100, out_features=10, bias=True)
      (1): BatchNorm1d(10, ...)
      (2): ReLU()
      (3): Dropout(p=0.5, inplace=False)
    )
    """
    layers = [
        layer,
        NORM[norm](c_out) if norm is not None else None,
        act() if act is not None else None,
        nn.Dropout(drop) if drop is not None else None]
    return nn.Sequential(*[x for x in layers if x is not None])


def make_dense(
        c_all: list[int],
        norm: bool,
        act: type = None,
        drop: float = None
):
    """Create a sequential neural network with dense layers.

    Parameters
    ----------
    c_all : list[int]
        Channels in each dense layer. Total network has len(channels) - 1
        layers, where layer[i] has channels[i] input and channels[i + 1]
        output channels.
    norm : bool
        If True, perform batch normalization.
    act : type, optional
        Passed to make_block function call.
        Defaults to None.
    drop : float, optional
        Passed to make_block function call.
        Defaults to None.

    Returns
    -------
    mlp : nn.Module
        Sequential neural network made of dense layers.

    Examples
    --------
    Dense network with normalization.
    >>> test_c_all = [10, 20, 5]
    >>> make_dense(test_c_all, norm=True)  # doctest: +ELLIPSIS
    Sequential(
      (0): Sequential(
        (0): Linear(in_features=10, out_features=20, bias=True)
        (1): BatchNorm1d(20, ...)
      )
      (1): Sequential(
        (0): Linear(in_features=20, out_features=5, bias=True)
        (1): BatchNorm1d(5, ...)
      )
    )

    Dense network with normalization, ReLu, and dropout layers.
    >>> make_dense(test_c_all, norm=True, act=nn.ReLU, drop=0.5)
    ... # doctest: +ELLIPSIS
    Sequential(
      (0): Sequential(
        (0): Linear(in_features=10, out_features=20, bias=True)
        (1): BatchNorm1d(20, ...)
        (2): ReLU()
        (3): Dropout(p=0.5, inplace=False)
      )
      (1): Sequential(
        (0): Linear(in_features=20, out_features=5, bias=True)
        (1): BatchNorm1d(5, ...)
        (2): ReLU()
        (3): Dropout(p=0.5, inplace=False)
      )
    )
    """
    norm = "1d" if norm else None
    mlp = [
        make_block(nn.Linear(c_all[i], c), c, norm=norm, act=act, drop=drop)
        for i, c in enumerate(c_all[1:])]
    mlp = mlp[0] if len(mlp) == 1 else nn.Sequential(*mlp)
    return mlp
