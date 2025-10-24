#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


# 1. Standard library imports
from pathlib import Path

# 2. Third-party library imports
import numpy as np
import zarr
from numcodecs import Blosc


def make_zarr(
        file: Path,
        shape: tuple[int, ...],
        chunk: tuple[int, ...],
        mode: str = "a",
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
    shape : tuple[int, ...]
        Shape of array to store in zarr file.
    chunk : tuple[int, ...]
        Shape of data chunks stored in zarr file.
    mode : {"r", "a", "w", "x"}, optional
        File open mode.
        Defaults to "a".
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
        file.with_suffix(".zarr"), mode=mode, shape=shape, chunks=chunk,
        dtype=dtype, fill_value=fill, compressor=compressor, **kwargs)
    return zarr_file


def resize_zarr(
        z_arr: zarr.Array,
        axes: dict[int, int]
):
    """Resize zarr array.

    Parameters
    ----------
    z_arr : zarr.Array
        Zarr array to resize.
    axes : dict[int, int]
        key : int
            Index of axis to resize.
        value:
            Quantity to extend specified axis.

    Returns
    -------
    old_shape : tuple[int]
        Shape of `zarr_file` before resizing.
    """
    old_shape = z_arr.shape
    new_shape = np.array(old_shape, dtype=int)
    for k, v in axes.items():
        new_shape[k] += v

    z_arr.resize(new_shape)
    return old_shape


def consolidate_zarr(
        z_arr: Path | zarr.Array
):
    """Work with consolidated zarr array metadata.

    Parameters
    ----------
    z_arr : Path | zarr.Array
        If ``Path``, open zarr file with consolidated metadata (.zarr).
        Otherwise, if ``zarr.Array``, consolidate file metadata.

    Returns
    -------
    z_arr : zarr.Array
        `z_arr` after manipulation
    """
    if isinstance(z_arr, Path):
        z_arr = zarr.open_consolidated(z_arr, mode="r")
    else:
        zarr.consolidate_metadata(z_arr.store)

    return z_arr
