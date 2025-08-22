#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July 4 13:06:21 2023
@author: ike
"""


import numpy as np
import scipy.stats as sst
import scipy.ndimage as scn
import skimage.morphology as skm


def label_array(
        arr: np.ndarray,
        connect: int = None
):
    """
    Label connected components in an array.

    Args:
    ---------------------------------------------------------------------------
        arr (np.ndarray):
            Array to label.
        connect (int, optional):
            Determine which array indices to connect as neighbors. 1 <= connect
            <= arr.ndim.
            Defaults to None, in which case connect = array.ndim.

    Returns:
    ---------------------------------------------------------------------------
        array (np.ndarray):
            Input array with connected components labeled (int).
        mode (int):
            Label of largest connected component.
        count (int):
            Size of largest connected component.

    Examples:
    ---------------------------------------------------------------------------
        Label an array with 2 objects connected by faces and vertices
        >>> array = np.array([
        ...     [0, 1, 0, 4, 0, 0, 0, 1, 0, 0],
        ...     [8, 7, 0, 0, 9, 0, 0, 2, 2, 0],
        ...     [0, 0, 0, 0, 9, 8, 8, 0, 0, 0]])
        >>> labeled, mode, count = label_array(array)
        >>> labeled
        array([[0, 1, 0, 2, 0, 0, 0, 2, 0, 0],
               [1, 1, 0, 0, 2, 0, 0, 2, 2, 0],
               [0, 0, 0, 0, 2, 2, 2, 0, 0, 0]], dtype=int32)
        >>> mode
        np.int32(2)
        >>> count
        np.int64(8)

        Label an array with 4 objects connected by faces alone
        >>> labeled, _, _ = label_array(array, connect=1)
        >>> labeled
        array([[0, 1, 0, 2, 0, 0, 0, 3, 0, 0],
               [1, 1, 0, 0, 4, 0, 0, 3, 3, 0],
               [0, 0, 0, 0, 4, 4, 4, 0, 0, 0]], dtype=int32)
    """
    connect = min(arr.ndim, arr.ndim if connect is None else connect)
    arr, _ = scn.label(
        arr, structure=scn.generate_binary_structure(arr.ndim, connect))
    mode, count = sst.mode(np.ma.masked_equal(arr, 0), axis=None)
    return arr, mode, count


def remove_blobs(
        arr: np.ndarray,
        min_size: int,
        connect: int = None
):
    """
    Remove foreground objects smaller than threshold size.

    Args:
    ---------------------------------------------------------------------------
        arr (np.ndarray):
            Array to apply volume threshold.
        min_size (int):
            Minimum volume of foreground object in pixels.
        connect (int, optional):
            Determine which array indices to connect as neighbors. Passed to
            label_array function call.

    Returns:
    ---------------------------------------------------------------------------
        (np.ndarray):
            Array with foreground objects removed (bool).

    Examples:
    ---------------------------------------------------------------------------
        Remove objects with fewer than 5 indices
        >>> array = np.array([
        ...     [0, 1, 0, 4, 0, 0, 0, 1, 0, 0],
        ...     [8, 7, 0, 0, 9, 0, 0, 2, 2, 0],
        ...     [0, 0, 0, 0, 9, 8, 8, 0, 0, 0]])
        >>> mask = remove_blobs(array, min_size=5)
        >>> mask * array
        array([[0, 0, 0, 4, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 9, 0, 0, 2, 2, 0],
               [0, 0, 0, 0, 9, 8, 8, 0, 0, 0]])
    """
    connect = arr.ndim if connect is None else min(connect, arr.ndim)
    arr, _, _ = label_array(arr, connect)
    arr = arr if np.unique(arr).size <= 2 else skm.remove_small_objects(
        arr, min_size, connect)
    return arr != 0


def min_max_scaling(
        arr: np.ndarray,
        new_min: float = 0.0,
        new_max: float = 1.0,
        dtype: np.dtype = None,
):
    """
    Rescale array to set minimum and maximum values.

    Args:
    ---------------------------------------------------------------------------
        arr (np.ndarray):
            Array to scale.
        new_min (float, optional):
            Minimum value in scaled array.
            Defaults to 0.
        new_max (float, optional):
            Maximum value in scaled array.
            Defaults to 1.
        dtype (np.dtype, optional):
            Data type of the scaled array.
            Defaults to None, in which case return type is np.float64.

    Returns:
    ---------------------------------------------------------------------------
        arr (np.ndarray):
            Array values scaled between new_min and new_max.
        old_min (float):
            Minimum value in input array.
        old_max (float):
            Maximum value in input array.

    Examples:
    ---------------------------------------------------------------------------
        Rescale integer array to [0, 4] interval
        >>> array = np.array([
        ...     [0, 1, 0, 4, 0],
        ...     [8, 7, 0, 0, 6],
        ...     [0, 0, 0, 0, 4]], dtype=int)
        >>> array, old_min, old_max = min_max_scaling(
        ...     array, new_min=0, new_max=4)
        >>> array
        array([[0. , 0.5, 0. , 2. , 0. ],
               [4. , 3.5, 0. , 0. , 3. ],
               [0. , 0. , 0. , 0. , 2. ]])
        >>> old_min
        np.int64(0)
        >>> old_max
        np.int64(8)
    """
    old_min, old_max = np.min(arr), np.max(arr)
    arr = (arr - old_min) / (old_max - old_min)
    arr = (arr * (new_max - new_min)) + new_min
    arr = arr if dtype is None else arr.astype(dtype)
    return arr, old_min, old_max


def aggregate_region(
        arr: np.ndarray,
        mask: np.ndarray,
        func: np.ufunc = np.mean
):
    """
    Computes some statistic from a region specified by the mask.

    Args:
    ---------------------------------------------------------------------------
        arr (np.ndarray):
            Array on which to compute statistic within a region.
        mask (np.ndarray):
            Boolean or binary array identifying region of interest within arr.
            Must either have the same shape as arr, or a shape compatible with
            the trailing dimensions of arr.
        func (np.ufunc, optional):
            Numpy-compatible function used to aggregate values within a region.
            Defaults to np.mean.

    Returns:
    ---------------------------------------------------------------------------
        (float | np.ndarray):
            Statistic from arr indices within masked region.
            Shape corresponds to leading dimensions of arr beyond those
            included in mask (shape = arr.shape[:mask.ndim]). If arr and mask
            have the same shape, return value is float.

    Examples:
    ---------------------------------------------------------------------------
        arr.ndim > mask.ndim
        >>> array = np.arange(30).reshape(3, 10)
        >>> array
        array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
        >>> mask = np.arange(10) >= 5
        >>> aggregate_region(array, mask)
        array([ 7., 17., 27.])

        arr.ndim = mask.ndim
        >>> aggregate_region(array[0], mask)
        7.0
    """
    arr = arr[..., mask]
    shape = arr.shape[:-1]
    arr = func(arr.reshape(-1, arr.shape[-1]), axis=-1)
    arr = arr.reshape(shape) if arr.size > 1 else arr.item()
    return arr


def project_arr(
        arr: np.ndarray,
        func: np.ufunc,
        step: int = 50
):
    """
    Generate projections along all axes of an array. Useful for array-like
    objects (h5py.dataset, zarr.Array, etc.) too large to fit in memory.

    Args:
    ---------------------------------------------------------------------------
        arr (np.ndarray):
            Array-like object to project.
        func (np.ufunc):
            numpy function with which to project along each axis.
        step (int, optional):
            Number of indices along the first dimension to process at a time.
            Defaults to 50.

    Returns:
    ---------------------------------------------------------------------------
        proj (list[np.ndarray]):
            Projection array along each axis of arr.

    Examples:
    ---------------------------------------------------------------------------
        Min intensity projection
        >>> array = np.arange(30).reshape(3, 10)
        >>> array
        array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
        >>> proj_0, proj_1 = project_arr(array, func=np.mean)
        >>> proj_0
        array([10., 11., 12., 13., 14., 15., 16., 17., 18., 19.])
        >>> proj_1
        array([ 4.5, 14.5, 24.5])
    """
    # if True, compute initial projection via slice-wise comparison
    slice_by_slice = func in (np.min, np.max)

    # project slices along z, y, and x dimensions
    proj = [
        np.zeros([d for j, d in enumerate(arr.shape) if j != i])
        for i in range(len(arr.shape))]

    # chunk along first dimension
    for idx in range(0, arr.shape[0], step):
        end = min(idx + step, arr.shape[0])
        sub = arr[idx:end]

        # compute projections on subset
        for dim in range(1, len(arr.shape)):
            proj[dim][idx:end] = func(sub, axis=dim)

        if slice_by_slice:
            val = func(sub, axis=0)
            proj[0] = val if idx == 0 else func(
                np.stack([proj[0], val], axis=0), axis=0)

    # update first projection if slice-by-slice comparison invalid
    if not slice_by_slice:
        proj[0] = np.concatenate([
            func(arr[:, ydx: min(ydx + step, arr.shape[1])], axis=0)
            for ydx in range(0, arr.shape[1], step)], axis=0)

    # return projections
    return proj


def dense_to_sparse(
        arr: np.ndarray
):
    """
    Extract coordinates of nonzero values in an array.

    Args:
    ---------------------------------------------------------------------------
        arr (np.ndarray):
            Dense array to convert to sparse format.

    Returns:
    ---------------------------------------------------------------------------
        coords (np.ndarray):
            Coordinates of nonzero values in arr. Has shape
            (nonzero_count, arr.ndim).
        values (np.ndarray):
            Corresponding nonzero values in arr. Has shape (nonzero_count,).

    Examples:
    ---------------------------------------------------------------------------
        Extract nonzero values from 3x10 array
        >>> array = np.arange(30).reshape(3, 10)
        >>> array[array % 9 != 0] = 0
        >>> array
        array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  9],
               [ 0,  0,  0,  0,  0,  0,  0,  0, 18,  0],
               [ 0,  0,  0,  0,  0,  0,  0, 27,  0,  0]])
        >>> coords, values = dense_to_sparse(array)
        >>> coords
        array([[0, 9],
               [1, 8],
               [2, 7]])
        >>> values
        array([ 9, 18, 27])
    """
    coords = np.nonzero(arr)
    coords, values = np.stack(list(coords), axis=-1), arr[*coords]
    return coords, values


def sparse_to_dense(
        coords: np.ndarray,
        values: np.ndarray,
        dims: list[int] | tuple[int] | np.ndarray[int] = None
):
    """
    Convert nonzero coordinates and values into a dense array.

    Args:
    ---------------------------------------------------------------------------
        coords (np.ndarray):
            Coordinates of nonzero values in dense array. Has shape
            (nonzero_count, arr.ndim).
        values (np.ndarray):
            Corresponding nonzero values in dense array. Has shape
            (nonzero_count,).
        dims (list[int] | tuple[int] | np.ndarray[int], optional):
            Shape of dense array.
            Defaults to None, in which case shape inferred from coordinates.

    Returns:
    ---------------------------------------------------------------------------
        dense (np.ndarray):
            Dense array. Dtype inferred from values.

    Examples:
    ---------------------------------------------------------------------------
        Dense size inferred from coordinates
        >>> arr = np.array([[0, 4], [1, 6], [0, 2]])
        >>> values = np.array([ 9, 18, 27])
        >>> sparse_to_dense(arr, values)
        array([[ 0,  0, 27,  0,  9,  0,  0],
               [ 0,  0,  0,  0,  0,  0, 18]])

        Dense size specified
        >>> dims = (3, 10)
        >>> sparse_to_dense(arr, values, dims)
        array([[ 0,  0, 27,  0,  9,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0, 18,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
    """
    dims = np.max(coords, axis=0) + 1 if dims is None else dims
    dense = np.zeros(np.asarray(dims, dtype=int), dtype=values.dtype)
    dense[tuple(coords.T.astype(int))] = values
    return dense
