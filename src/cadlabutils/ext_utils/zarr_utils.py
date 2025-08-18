#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import zarr
import numpy as np

from pathlib import Path
from numcodecs import Blosc


"""
zarr files
===============================================================================
"""


def make_zarr(
        file: Path,
        shape: tuple,
        chunk: tuple,
        dtype: type = np.uint8,
        fill: int = 0,
        compressor: type = None,
        **kwargs
):
    """
    Create zarr file with set fill value.

    NOTE: this function will overwrite existing zarr file with same path.
    Args:
        file (Path):
            Path to save zarr file (.zarr).
        shape (tuple):
            Shape of array stored in zarr file.
        chunk (tuple):
            Shape of data chunks stored in zarr file.
        dtype (type, optional):
            Datatype of saved array.
            Defaults to np.uint8.
        fill (int, optional):
            Default value of array indices not written to zarr file.
            Defaults to 0.
        compressor (type, optional):
            Dile compression. Passed to zarr.open function call.
            Defaults to Blosc.
        **kwargs:
            Keyword arguments passed to zarr.open function call.

    Returns:
        zarr_file (zarr.Array):
            Instantiated zarr file.
    """
    compressor = compressor if compressor is not None else Blosc(
        cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    zarr_file = zarr.open(
        file, mode="w", shape=shape, chunks=chunk, dtype=dtype,
        fill_value=fill, compressor=compressor, **kwargs)
    return zarr_file


def resize_zarr(
        zarr_file: zarr.Array,
        add: list[int]
):
    """
    Resize zarr array.

    Args:
        zarr_file (zarr.Array):
            Zarr array to resize.
        add (list[int]):
            Extension to add to each dimension of zarr array. Index corresponds
            to dimension in array.

    Returns:
        zarr_file (zarr.Array):
            Resized zarr array.
        start (list[int]):
            Shape of array before resizing.
    """
    start = [0 if s is None else s for s in zarr_file.shape]
    new_size = [
        s + (add[i] if i < len(add) else 0) for i, s in enumerate(start)]
    zarr_file.resize(new_size)
    return zarr_file, start


def project_zarr(
        zarr_file: zarr.Array,
        func: np.ufunc,
        axes: str = "zyx",
        step: int = 50
):
    """
    Generate projections along all axes of a z-stack stored in zarr format.

    Args:
        zarr_file (zarr.Array):
            Zarr array to project.
        func (np.ufunc, optional):
            numpy function with which to project along each axis.
            Defaults to np.min.
        step (int, optional):
            Number of z-planes to process at a time. Defaults to 50.

    Returns:
        projections (dict[str: np.ndarray]):
            Contains the following structure:
            -   key (str):
                    Projection axis. "z", "y", "x".
            -   value (np.ndarray):
                    Projection array.
    """
    # load data
    with h5.File(file, "r") as data:
        dset = data[name]

        # project slices along z, y, and x dimensions
        z_proj = None
        y_proj, x_proj = np.zeros(dset.shape[::2]), np.zeros(dset.shape[:-1])
        for zdx in range(0, dset.shape[0], step):
            end = min(zdx + step, dset.shape[0])
            sub = dset[zdx:end]

            # compute projections on subset
            y_proj[zdx:end] = func(sub, axis=1)
            x_proj[zdx:end] = func(sub, axis=-1)
            val = func(sub, axis=0)
            if func == np.mean:
                val *= (sub.shape[0] / dset.shape[0])
                z_proj = val if z_proj is None else z_proj + val
            else:
                z_proj = val if z_proj is None else func(
                    np.stack([z_proj, val], axis=0), axis=0)

    # return projections
    return {"z": z_proj, "y": y_proj, "x": x_proj}