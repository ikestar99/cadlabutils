#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import hdf5plugin

from h5py import File, Group
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
        shape: tuple[int],
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
    shape : tuple[int]
        Shape of dataset.
    dtype : type | str
        Data type of dataset.
    fill : float | str, optional
        Fill value for unwritten data.
        Defaults to 0.0.

    Returns
    -------
    dataset (h5.Dataset):
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
