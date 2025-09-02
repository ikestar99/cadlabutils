#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import zarr
import numpy as np

from typing import Union
from pathlib import Path
from numcodecs import Blosc


IntArrayLike = Union[list[int] | tuple[int] | np.ndarray[int]]


def make_zarr(
        file: Path,
        shape: tuple[int],
        chunk: tuple[int],
        dtype: type = np.uint8,
        fill: float = 0.0,
        compressor: type = "blosc",
        **kwargs
):
    """Create zarr file with set fill value.

    Parameters
    ----------
    file : Path
        Path to save zarr file (.zarr).
    shape : tuple[int]
        Shape of array to store in zarr file.
    chunk : tuple[int]
        Shape of data chunks stored in zarr file.
    dtype : type, optional
        Datatype of saved array.
        Defaults to np.uint8.
    fill : float, optional
        Default value of array indices not written to zarr file.
        Defaults to 0.0.

    Returns
    -------
    zarr_file : zarr.Array
        Opened zarr file.

    Other Parameters
    ----------------
    compressor : optional
        File compression. Passed to zarr.open function call.
        Defaults to "blosc".
    **kwargs : dict
        Keyword arguments passed to zarr.open.
    """
    compressor = compressor if compressor != "blosc" else Blosc(
        cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    zarr_file = zarr.open(
        file, mode="w", shape=shape, chunks=chunk, dtype=dtype,
        fill_value=fill, compressor=compressor, **kwargs)
    return zarr_file


def resize_zarr(
        zarr_file: zarr.Array,
        new_shape: IntArrayLike
):
    """Resize zarr array.

    Parameters
    ----------
    zarr_file : zarr.Array
        Zarr array to resize.
    new_shape : IntArrayLike
        Shape of `zarr_file` after resizing.

    Returns
    -------
    zarr_file : zarr.Array
        Resized `zarr_file`.
    """
    zarr_file.resize(np.maximum(zarr_file.shape, new_shape))
    return zarr_file
