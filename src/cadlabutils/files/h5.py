#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import h5py as h5
import numpy as np
import hdf5plugin

from pathlib import Path


"""
Processing data in arrays
===============================================================================
"""


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
        return data[h5_dir].shape


# def print_hdf5_structure(file_path):
#     """Prints the structure of an HDF5 file."""
#     with h5.File(file_path, 'r') as hf:
#         def print_item(name, obj):
#             tabs = name.count("/")
#             start = name.rindex("/") + 1 if tabs > 0 else 0
#             shape = obj.shape if isinstance(obj, h5.Dataset) else ""
#             print(f"{"    " * tabs}{name[start:]} {shape}")
#
#         print(f"Structure of HDF5 file: {file_path}")
#         hf.visititems(print_item)

