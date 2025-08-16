#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import zarr
import h5py as h5
import numpy as np
import hdf5plugin

from pathlib import Path
from numcodecs import Blosc


"""
Processing data in arrays
===============================================================================
"""


def safe_concat(
        arr_1: np.ndarray,
        arr_2: np.ndarray = None,
        axis: int = 0
):
    """
    Concatenate two arrays along a specified axis if neither is None. Used to
    abstract the following code:
    arr = concatenate((arr_2, arr_1), axis) if arr_2 is not None else arr_1

    Args:
        arr_1 (np.ndarray):
            First array to concatenate.
        arr_2 (np.ndarray, optional):
            Second array to concatenate.
            Defaults to None, in which case output is arr_1.
        axis (int, optional):
            Axis along which to concatenate.
            Defaults to 0.

    Returns:
        (np.ndarray):
            Concatenated array.
    """
    if arr_2 is not None:
        return np.concatenate((arr_2, arr_1), axis=axis)

    return arr_1


def make_h5_dataset(
        base: h5.File | h5.Group,
        name: str,
        shape: tuple,
        dtype: type = np.uint8
):
    """
    Create dataset within h5 file.

    Args:
        base (h5.File | h5.Group):
            Location in which to create a dataset.
        name (str):
            Name of dataset.
        shape (tuple):
            Shape of dataset.
        dtype (str, optional):
            Datatype of saved array.
            Defaults to "uint8".

    Returns:
        dataset (h5.Dataset):
            Instantiated dataset.
    """
    dataset = base.create_dataset(
        name, shape=shape, chunks=True, dtype=dtype, fillvalue=0,
        **hdf5plugin.Blosc(
            cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
    return dataset


def get_h5_dtype(
        data_h5: Path,
        h5_dir: str
):
    """
    Extract datatype of dataset in hdf5 file.

    Args:
        data_h5 (Path):
            Path to stack data (h5).
        h5_dir (str):
            Path to dataset within h5 file.

    Returns:
        dtype (type):
            Datatype of dataset.
    """
    with h5.File(data_h5, "r") as data:
        dtype = data[h5_dir].dtype

    return dtype


def get_h5_shape(
        data_h5: Path,
        h5_dir: str
):
    """
    Extract shape of dataset in hdf5 file.

    Args:
        data_h5 (Path):
            Path to stack data (h5).
        h5_dir (str):
            Path to dataset within h5 file.

    Returns:
        shape (tuple[int, ...):
            Shape of dataset.
    """
    with h5.File(data_h5, "r") as data:
        shape = data[h5_dir].shape

    return shape


def project_h5(
        file: Path,
        name: str,
        func: np.ufunc = np.min,
        step: int = 50
):
    """
    Generate projections along all axes of a z-stack stored in zarr format.

    Args:
        file (Path):
            Path to z-stack to project (h5). Of shape (z, y, x).
        name (str):
            Path to z-stack dataset within h5 file.
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


def make_zarr(
        file: Path,
        shape: tuple,
        chunk: tuple,
        dtype: type = np.uint8,
        fill: int = 0
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
        shard (tuple, optional):
            Size of each individual file written to disk.
            Defaults to None, in which case shards are not used.
        max_shape (tuple, optional).
            Maximum shape of array stored in zarr file, if resized on the fly.
            Defaults to None, in which case maxshape is equal to shape.
        dtype (type, optional):
            Datatype of saved array.
            Defaults to np.uint8.
        fill (int, optional):
            Default value of array indices not written to zarr file.
            Defaults to 0.

    Returns:
        (zarr.Array):
            Instantiated zarr file.
    """
    zarr_file = zarr.open(
        file, mode="w", shape=shape, chunks=chunk, dtype=dtype,
        fill_value=fill, compressor=Blosc(
            cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE))
    return zarr_file
