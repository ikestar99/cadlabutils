#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import numpy as np
import hdf5plugin

from h5py import File, Group, Dataset
from pathlib import Path


def get_metadata(
        file: Path,
        h5_dir: str
):
    """Extract shape and data type of hdf5 dataset.

    Parameters
    ----------
    file : Path
        Path to stored data (.h5).
    h5_dir : str
        Path to dataset within h5 file.

    Returns
    -------
    tuple
        Shape of dataset at `h5_dir`.
    type
        Data type of dataset.
    """
    with File(file, "r") as data:
        dset = data[h5_dir]
        return dset.shape, dset.dtype


def get_tree(
        file: Path
):
    """Extract structure of a hdf5 file from available metadata.

    Parameters
    ----------
    file : Path
        Path to hdf5 file (.h5, .hdf5).

    Returns
    -------
    tree : dict[dict]
        Structure of stored data. Keys are names of nested groups and datasets.
        For datasets, value is another dictionary with the following structure:
        -   "shape": tuple of ints, shape of dataset.
        -   "dtype": data type of dataset.

    Examples
    --------
    >> from ... import print_tree


    >> test_tree = get_tree(Path(".../file.h5"))
    >> print_tree(test_tree, color=False)
    data
    ├── mask
    │   ├── shape: (1, 147, 5797, 3921)
    │   └── dtype: uint8
    └── raw
        ├── shape: (1, 147, 5797, 3921)
        └── dtype: uint8
    """
    def _visit(name, node, tree):
        parts = name.split('/')
        current = tree
        for part in parts[:-1]:
            current = current.setdefault(part, {})

        if isinstance(node, Group):
            current.setdefault(parts[-1], {})
        else:
            current[parts[-1]] = {
                "shape": tuple(node.shape),
                "dtype": node.dtype}

    tree = {}
    with File(file, "r") as data:
        data.visititems(lambda name, node: _visit(name, node, tree))

    return tree


def make_dataset(
        base: File | Group,
        name: str,
        shape: tuple[int, ...],
        dtype: type,
        fill: float | str = 0.0,
        use_blosc: bool = True,
        **kwargs
):
    """Create dataset within h5 file.

    Parameters
    ----------
    base : File | Group
        Location in which to create a dataset.
    name : str
        Name of dataset.
    shape : tuple[int, ...]
        Shape of dataset.
    dtype : type | str
        Data type of dataset.
    fill : float | str, optional
        Fill value for unwritten data.
        Defaults to 0.0.

    Returns
    -------
    dataset : Dataset
        Instantiated dataset.

    Other Parameters
    ----------------
    use_blosc : bool, optional
        If True, use Blosc compression from hdf5plugin.
        Defaults to True.
    **kwargs : dict
        Keyword arguments passed to base.create_dataset.
    """
    compress = {} if not use_blosc else hdf5plugin.Blosc(
        cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
    dataset = base.create_dataset(
        name, shape=shape, dtype=dtype, chunks=True, fillvalue=fill,
        **compress, **kwargs)
    return dataset


def get_mean_std(
        dset: Dataset,
        axis: int = 0,
        step: int = 50
):
    """Apply two-pass algorithm to compute mean and std of large dataset.

    Parameters
    ----------
    dset : Dataset
        Dataset for which to compute mean and std.
    axis : int, optional
        Axis along which to chunk `dset`.
        Defaults to 0.
    step : int, optional
        Number of indices along `axis` to process as a time.
        Defaults to 50.

    Returns
    -------
    mean : float
        Mean value in `dset`.
    std : float
        Standard deviation of values in `dset`.
    """
    # First pass: mean
    total_sum, total_count = 0.0, 0
    for i in range(0, dset.shape[axis], step):
        indices = [slice(None) for _ in dset.shape]
        indices[axis] = slice(i, min(i + step, dset.shape[axis]))
        chunk = dset[tuple(indices)]
        total_sum += np.sum(chunk)
        total_count += chunk.size

    mean = total_sum / total_count

    # Second pass: variance
    total_sq_diff = 0.0
    for i in range(0, dset.shape[axis], step):
        indices = [slice(None) for _ in dset.shape]
        indices[axis] = slice(i, min(i + step, dset.shape[axis]))
        chunk = dset[tuple(indices)]
        total_sq_diff += np.sum((chunk - mean) ** 2)

    std = np.sqrt(total_sq_diff / (total_count - 1))
    return mean, std