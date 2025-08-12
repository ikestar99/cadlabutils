#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import os
import time
import h5py as h5
import numpy as np
import pandas as pd
import shutil
import platform
import tifffile as tf
import hdf5plugin
import tracemalloc as tm
import scipy.stats as sst
import rich.progress as rp
import scipy.spatial as ssp
import scipy.ndimage as scn
import skimage.morphology as skm

from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cdist

from src.cadlabutils.unionfind import UnionFind


UNITS = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"]






def save_tif(
        array: np.ndarray,
        file: Path,
        axes: str,
        dtype: type = np.uint8,
):
    """
    Save array as tif image.

    Args:
        array (ndarray):
            Array to save as tif.
        file (Path):
            Path to tif save location.
        axes (str):
            Axis order of tif image or hyperstack. 2D image is "YX", z stack is
            "ZYX". Saved as tif ImageJ metadata.
        dtype (type, optional):
            Desired datatype of saved array.
            Defaults to np.uint8.
    """
    tf.imwrite(
        file, array.astype(dtype), imagej=True, metadata={"axes": axes},
        compression="zlib")


"""
Convenient index slicing
===============================================================================
"""
def s_slice(
        start: int,
        stop: int,
        buffer: int,
        step: int = 1,
):
    """
    Create safe slice object with flexible behavior at start.

    Args:
        start (int):
            First index in slice.
        stop (int):
            Final index in slice.
        buffer (int):
            Quantity by which to lower beginning of slice.
        step (int):
            Interval between consecutive indices in slice.
            Defaults to 1.
    """
    return slice(max(start - buffer, 0), stop, step)


def e_slice(
        start: int,
        stop: int,
        buffer: int,
        step: int = 1,
):
    """
    Create safe slice object with flexible behavior at stop (end).

    Args:
        start (int):
            First index in slice.
        stop (int):
            Final index in slice.
        buffer (int):
            Quantity by which to extend end of slice.
        step (int):
            Interval between consecutive indices in slice.
            Defaults to 1.
    """
    return slice(start, min(start + buffer, stop), step)


def d_slice(
        start: int,
        stop: int,
        buffer: int,
        step: int = 1,
):
    """
    Create safe slice object with flexible behavior at start and stop.

    Args:
        start (int):
            First index in slice.
        stop (int):
            Final index in slice.
        buffer (int):
            Quantity by which to lower beginning and extend end of slice.
        step (int):
            Interval between consecutive indices in slice.
            Defaults to 1.
    """
    return slice(max(start - buffer, 0), min(start + buffer, stop), step)


"""
Geometry
===============================================================================
"""
def r_to_a(
        r: float
):
    """
    Compute area of circle with specified radius.

    Args:
        r (float):
            Radius of circle.

    Returns:
        (int):
            Area of circle.
    """
    return int(round(np.pi * (r ** 2)))


def r_to_v(
        r: float
):
    """
    Compute volume of sphere with specified radius.

    Args:
        r (float):
            Radius of sphere.

    Returns:
        (int):
            Volume of sphere.
    """
    return int(round((4 * np.pi * (r ** 3)) / 3))


"""
Array normalization
===============================================================================
"""
def min_max_scaling(
        array: np.ndarray,
        new_min: float = 0,
        new_max: float = 1,
):
    """
    Rescale array to set minimum and maximum values.

    Args:
        array (np.ndarray):
            Array to scale.
        new_min (float, optional):
            Minimum value in scaled array.
            Defaults to 0.
        new_max (float, optional):
            Maximum value in scaled array.
            Defaults to 1.

    Returns:
        (tuple):
            Contains the following three items:
            -   0 (np.ndarray):
                Array values scaled between new_min and new_max.
            -   1 (float):
                Minimum value in input array.
            -   2 (float):
                Maximum value in input array.
    """
    min_value, max_value = np.min(array), np.max(array)
    array = (array - min_value) / (max_value - min_value)
    array = (array * (new_max - new_min)) + new_min
    return array, min_value, max_value



"""
Sparse and dense matrix interconversion
===============================================================================
"""
def dense_to_sparse(
        dense: np.ndarray,
        axes: list,
        value: str,
        offset: list=None
):
    """
    Convert dense array into sparse DataFrame. Each nonzero element in array
    becomes a unique row in DataFrame. DataFrame has a column for each axis in
    array and an additional column for the value of each nonzero element.

    Args:
        dense (np.ndarray):
            Dense array to convert to sparse format.
        axes (list):
            String labels of coordinate columns in sparse DataFrame. Must have
            index for each axis in dense array.
        value (str):
            Label of element value column in sparse DataFrame.
        offset (list, optional):
            Additive offset to coordinate values in sparse DataFrame. Must have
            integer value for each axis in dense array.
            Defaults to None, in which case no offset is added.

    Returns:
        (pd.DataFrame):
            Sparse DataFrame. Has columns = axes + [value]
    """
    sparse = np.nonzero(dense)
    sparse = np.stack(list(sparse) + [dense[sparse]], axis=-1)
    sparse[..., :-1] += 0 if offset is None else np.array(offset)[np.newaxis]
    return pd.DataFrame(sparse, columns=axes + [value])


def sparse_to_dense(
        sparse: pd.DataFrame,
        axes: list,
        value: str,
        dtype: type = np.uint8,
        dims: np.ndarray = None
):
    """
    Convert sparse DataFrame into dense format. Dense array has an axis for
    each specified column in DataFrame with value populated by according to
    specified column. Background values set to 0.

    Args:
        sparse (pd.DataFrame):
            Sparse DataFrame to convert to dense format.
        axes (list):
            String labels of coordinate columns in sparse DataFrame. Must have
            index for each axis in dense array.
        value (str):
            Label of element value column in sparse DataFrame.
        dtype (type, optional):
            Data type of dense array.
            Defaults to np.uint8.
        dims (list, optional):
            Shape of dense array.
            Defaults to None, in which case shape inferred from coordinates.

    Returns:
        (np.ndarray):
            Dense array.
    """
    sparse = sparse[axes + [value]].to_numpy().T
    dims = np.max(sparse[:-1], axis=-1) if dims is None else dims
    dense = np.zeros(dims.astype(int) + 1, dtype=dtype)
    dense[tuple(sparse[:-1].astype(int))] = sparse[-1]
    return dense


"""
Connected component analysis
===============================================================================
"""
def get_struct(
        ndim: int,
        connect: int = None,
        asarray: bool = False
):
    """
    Generate structuring element for spatial operations on arrays.

    Args:
        ndim (int):
            Number of dimensions in array of interest.
        connect (int, optional):
            Define local connectivity of each pixel.
            - 1: only pixels connected by faces are considered neighbors.
            - 2: pixels connected by faces and edges are considered neighbors.
            - 3: pixels connected by faces, edges, and vertices are neighbors.
            Defaults to None, in which case connect = ndim.
        asarray (bool, optional):
            If True, generate an array structuring element.
            Defaults to False, in which case connect is returned.

    Returns:
        (int or np.ndarray):
            Connectivity rule with which to apply spatial operation on array.
    """
    struct = ndim if connect is None else min(connect, ndim)
    struct = scn.generate_binary_structure(ndim, struct) if asarray else struct
    return struct


def get_round(
        ndim: int,
        radius: int
):
    """
    Generate a spherical structuring element for an array of arbitrary
    dimensionality.

    Args:
        ndim (int):
            Number of dimensions in array of interest.
        radius (int):
            Radius of spherical structure.

    Returns:
        (np.ndarray):
            Spherical structuring element.
    """
    struct = np.zeros([2 * radius + 1] * ndim)
    struct[tuple([radius] * ndim)] = 0
    struct = scn.distance_transform_edt(struct) <= radius
    return struct.astype(int)


def label_array(
        array: np.ndarray,
        connect: int = None
):
    """
    Label connected components in an array.

    Args:
        array (np.ndarray):
            Array to label.
        connect (int):
            Determine which array indices to connect as neighbors.
            Defaults to None, in which case connect = array.ndim.

    Returns:
        (tuple):
            Contains the following three values:
            -   array (np.ndarray):
                    Input array with connected components labeled (int).
            -   mode (int):
                    Label of largest connected component.
            -   count (int):
                    Size of largest connected component.
    """
    array, _ = scn.label(
        array, structure=get_struct(array.ndim, connect, True))
    mode, count = sst.mode(np.ma.masked_equal(array, 0), axis=None)
    return array, mode, count


def remove_blobs(
        array: np.ndarray,
        volume: int,
        connect: int = None
):
    """
    Remove foreground objects smaller in area than volume threshold.

    Args:
        array (np.ndarray):
            Array to apply volume threshold.
        volume (int):
            Minimum volume of foreground object in pixels.
        connect (int):
            Determine which array indices to connect as neighbors.
            Defaults to None, in which case connect = array.ndim.

    Returns:
        (np.ndarray):
            Array with foreground objects removed (bool).
    """
    array, _, _ = label_array(array, connect)
    array = array if np.unique(array).size <= 2 else skm.remove_small_objects(
        array, volume, get_struct(array.ndim, connect))
    return array != 0


def grey_dilation(
        array: np.ndarray,
        radius: int
):
    """
    Merge discontinuities in foreground objects using greyscale dilation.

    Args:
        array (np.ndarray):
            Array on which to perform foreground object dilation.
        radius (int):
            radius of spherical kernel with which to perform dilation.

    Returns:
        (np.ndarray):
            Array with objects dilated. Same dtype as input.
    """
    dilate_array = scn.grey_dilation(
        array.astype(np.float32), footprint=get_round(array.ndim, radius))
    return dilate_array.astype(array.dtype)


def find_neighbors(
    coordinates: np.ndarray
):
    """
    Apply cKDTree to identify indices of neighboring points. Neighboring nodes
    are directly connected by faces, vertices, or edges -- approximately within
    a radius equal to the magnitude of the vector <1, ..., coordinates.ndim>.

    Args:
        coordinates (np.ndarray):
            Input array of point coordinates. Of shape (n_points, n_axes).

    Returns:
        (np.ndarray):
            Array of lists with the same length as coordinates arg. Each index
            contains a list of indices of all points connected to the current
            point, inclusive of the queried point itself.
    """
    tree = ssp.cKDTree(coordinates)
    neighbors = np.array(
        tree.query_ball_tree(tree, r=coordinates.shape[-1] + 0.1), dtype=list)
    return neighbors


def find_components(
        coordinates: np.ndarray,
        center: np.ndarray
):
    """
    Identify connected components from array of node coordinates and assign
    parent/child relationships between adjacent nodes. Will call find_neighbors
    to identify indices of neighboring points. Root node of each connected
    component is node closest to center point.

    Args:
        coordinates (np.ndarray):
            Input array of point coordinates. Of shape (n_points, n_axes).
        center (np.ndarray):
            Reference point. Of shape (n_axes,).

    Returns:
        (tuple):
            Contains 4 items, all 1d arrays of same length as number of nodes:
            -   0 (np.ndarray):
                    Label of connected component to which each node belongs.
            -   1 (np.ndarray):
                    Parent node index of each node.
            -   2 (np.ndarray):
                    Number of child nodes connected to each root node.
            -   3 (np.ndarray):
                    Euclidean distance of each node from center point.
    """
    uf = UnionFind(coordinates.shape[0])
    neighbors = find_neighbors(coordinates)
    distances = cdist(center[None], coordinates).flatten()
    for x in np.argsort(distances):
        for n in neighbors[x]:
            if n != x:
                uf.union(x, n)

    return *uf.update(), distances


"""
h5py and zarr backend
===============================================================================
"""
def make_h5_dataset(
        base: h5.File | h5.Group,
        name: str,
        shape: tuple,
        dtype: type=np.uint8
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
