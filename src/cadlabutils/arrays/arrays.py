#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July 4 13:06:21 2023
@author: ike
"""


# 1. Standard library imports
from typing import Callable

# 2. Third-party library imports
import numpy as np
import scipy.ndimage as scn
import scipy.stats as sst
from scipy.spatial.distance import pdist
import skimage.morphology as skm


rng = np.random.default_rng(42)


def get_arr_repeat(
        arr: np.ndarray,
        target: float,
        levels: int = 0
):
    """Compute number of arrays that fit in a fixed memory volume.

    Parameters
    ----------
    arr : np.ndarray
        Array to tile.
    target : float
        Size of memory container.
    levels : int, optional
        Unit of `target`. 0, B; 1, KiB; 2, MiB; 3, GiB, etc.
        Defaults to 0.

    Returns
    -------
    rep : int
        Number of `arr` instances that fit into the memory volume defined by
        `target` and `levels`.

    Examples
    --------
    >>> t_rep = get_arr_repeat(
    ...     np.ones((10, 10, 10), dtype=np.uint8), target=100, levels=1)
    >>> t_rep
    102
    >>> t_rep = get_arr_repeat(
    ...     np.ones((10, 10, 10), dtype=np.float64), target=100, levels=1)
    >>> t_rep
    12
    """
    rep = (target * (1024 ** levels)) // arr.nbytes
    return rep


def label_arr(
        arr: np.ndarray,
        connect: int = None
):
    """Label connected components in an array.

    Parameters
    ----------
    arr : np.ndarray
        Array to label.
    connect : int, optional
        Determine which relative indices to connect as neighbors. 1 --> connect
        by faces, 2 --> connect by edges, etc.
        Defaults to None, in which case assume dense connections.

    Returns
    -------
    labeled : np.ndarray[int]
        Input array with connected components labeled.
    n_labels : int
        Number of connected components.
    mode : int
        Label of largest connected component.
    count : int
        Size of largest connected component.

    Examples
    --------
    Label an array with 2 objects connected by faces and vertices
    >>> t_arr = np.array([
    ...     [0, 1, 0, 4, 0, 0, 0, 1, 0, 0],
    ...     [8, 7, 0, 0, 9, 0, 0, 2, 2, 0],
    ...     [0, 0, 0, 0, 9, 8, 8, 0, 0, 0]])
    >>> t_label, t_n_labels, t_mode, t_count = label_arr(t_arr)
    >>> t_label
    array([[0, 1, 0, 2, 0, 0, 0, 2, 0, 0],
           [1, 1, 0, 0, 2, 0, 0, 2, 2, 0],
           [0, 0, 0, 0, 2, 2, 2, 0, 0, 0]], dtype=int32)
    >>> t_n_labels
    2
    >>> t_mode
    np.int32(2)
    >>> t_count
    np.int64(8)

    Label an array with 4 objects connected by faces alone
    >>> t_label, _, _, _ = label_arr(t_arr, connect=1)
    >>> t_label
    array([[0, 1, 0, 2, 0, 0, 0, 3, 0, 0],
           [1, 1, 0, 0, 4, 0, 0, 3, 3, 0],
           [0, 0, 0, 0, 4, 4, 4, 0, 0, 0]], dtype=int32)
    """
    connect = min(arr.ndim, arr.ndim if connect is None else connect)
    labeled, n_labels = scn.label(
        arr, structure=scn.generate_binary_structure(arr.ndim, connect))
    mode, mode_count = (
        sst.mode(np.ma.masked_equal(labeled, 0), axis=None)
        if n_labels != 0 else (None, None))
    return labeled, n_labels, mode, mode_count


def remove_blobs(
        arr: np.ndarray,
        min_size: int,
        connect: int = None
):
    """
    Remove foreground objects smaller than threshold size.

    Parameters
    ----------
    arr : np.ndarray
        Array to apply volume threshold.
    min_size : int
        Minimum volume of foreground object in pixels.
    connect : int, optional
        Determine which array indices to connect as neighbors. Passed to
        label_arr function call.

    Returns
    -------
    arr : np.ndarray
        Input array with small objects removed.

    Examples
    --------
    Remove objects with fewer than 5 indices
    >>> t_arr = np.array([
    ...     [0, 1, 0, 4, 0, 0, 0, 1, 0, 0],
    ...     [8, 7, 0, 0, 9, 0, 0, 2, 2, 0],
    ...     [0, 0, 0, 0, 9, 8, 8, 0, 0, 0]])
    >>> remove_blobs(t_arr, min_size=5)
    array([[0, 0, 0, 4, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 9, 0, 0, 2, 2, 0],
           [0, 0, 0, 0, 9, 8, 8, 0, 0, 0]])
    """
    connect = arr.ndim if connect is None else min(connect, arr.ndim)
    labeled, n_labels, _, _ = label_arr(arr, connect)
    labeled = labeled if n_labels == 0 else skm.remove_small_objects(
        labeled, min_size, connect)
    return arr * (labeled != 0)


def min_max_scaling(
        arr: np.ndarray,
        new_min: float = 0.0,
        new_max: float = 1.0,
        dtype: np.dtype = None,
):
    """Rescale array to set minimum and maximum values.

    Parameters
    ----------
    arr : np.ndarray
        Array to scale.
    new_min : float, optional
        Minimum value in scaled array.
        Defaults to 0.
    new_max : float, optional
        Maximum value in scaled array.
        Defaults to 1.
    dtype : np.dtype, optional
        Data type of the scaled array.
        Defaults to None, in which case return type same as `arr` dtype.

    Returns
    -------
    scaled : np.ndarray
        `arr` values scaled between new_min and new_max.
    old_min : float
        Minimum value in `arr`.
    old_max : float
        Maximum value in `arr`.

    Examples
    --------
    Rescale integer array to [0, 4] interval
    >>> t_arr = np.array([
    ...     [0, 1, 0, 4, 0],
    ...     [8, 7, 0, 0, 6],
    ...     [0, 0, 0, 0, 4]], dtype=int)
    >>> t_arr, t_min, t_max = min_max_scaling(
    ...     t_arr, new_min=0, new_max=4)
    >>> t_arr
    array([[0. , 0.5, 0. , 2. , 0. ],
           [4. , 3.5, 0. , 0. , 3. ],
           [0. , 0. , 0. , 0. , 2. ]])
    >>> t_min
    np.int64(0)
    >>> t_max
    np.int64(8)
    """
    dtype = arr.dtype if dtype is None else dtype
    old_min, old_max = np.min(arr), np.max(arr)
    scaled = (arr - old_min) / (old_max - old_min)
    scaled = (scaled * (new_max - new_min)) + new_min
    return scaled.astype(dtype), old_min, old_max


def dtype_norm(
        arr: np.ndarray,
        offset: float = None
):
    """Normalize array to max value allowed by dtype.

    Parameters
    ----------
    arr : np.ndarray
        Array to normalize.
    offset : float, optional
        Normalize and subtract reference value from scaled array.
        Defaults to None, in which case no offset is performed.

    Returns
    -------
    normed : np.ndarray
        `arr` values normalized between [-1, 1] where 1 corresponds to the
        maximum value allowed by `arr` dtype.

    Raises
    ------
    NotImplementedError
        Attempts to normalize non-numeric arrays.

    Examples
    --------
    Normalize 8-bit array.
    >>> t_arr = (2 ** np.arange(3, 8)).astype(np.uint8)
    >>> t_arr
    array([  8,  16,  32,  64, 128], dtype=uint8)
    >>> dtype_norm(t_arr)
    array([0.03137255, 0.0627451 , 0.1254902 , 0.25098039, 0.50196078])

    Offset normalized array by 32.
    >>> dtype_norm(t_arr, offset=32)
    array([-0.09411765, -0.0627451 ,  0.        ,  0.1254902 ,  0.37647059])

    Normalize float array.
    >>> t_arr = np.array([-65500, -3000, 0, 3000, 65500]).astype(np.float16)
    >>> dtype_norm(t_arr)
    array([-1.    , -0.0458,  0.    ,  0.0458,  1.    ], dtype=float16)
    """
    info = np.iinfo if np.issubdtype(arr.dtype, np.integer) else np.finfo
    try:
        peak = max(abs(info(arr.dtype).min), info(arr.dtype).max)
        normed = arr / peak
        normed = normed - (offset / peak) if offset is not None else normed
        return normed
    except ValueError:
        raise NotImplementedError(f"Cannot normalize {arr.dtype} array.")


def masked_statistic(
        arr: np.ndarray,
        mask: np.ndarray,
        func: Callable = np.mean
):
    """Compute statistic from a region specified by a mask.

    Parameters
    ----------
    arr : np.ndarray
        Array on which to compute statistic within a region.
    mask : np.ndarray
        Boolean or binary array identifying region of interest within `arr`.
        Shape must match trailing dimensions of `arr`.
    func : Callable, optional
        Numpy-compatible function used to aggregate values within a region.
        Must accept an `axis` keyword argument.
        Defaults to np.mean.

    Returns
    -------
    region : float | np.ndarray
        Statistic from `arr` indices within masked region. Shape corresponds to
        leading dimensions of `arr`. If `arr` and `mask` have the same shape,
        return value is ``float``.

    Examples
    --------
    arr.ndim > mask.ndim
    >>> t_arr = np.arange(30).reshape(3, 10)
    >>> t_arr
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
    >>> t_mask = np.arange(10) >= 5
    >>> t_mask.astype(int)
    array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> masked_statistic(t_arr, t_mask)
    array([ 7., 17., 27.])

    arr.ndim = mask.ndim
    >>> masked_statistic(t_arr[0], t_mask)
    7.0
    """
    region = arr[..., mask]
    shape = region.shape[:-1]
    region = func(region.reshape(-1, region.shape[-1]), axis=-1)
    region = region.reshape(shape) if region.size > 1 else region.item()
    return region


def project_arr(
        arr: np.ndarray,
        func: Callable,
        step: int = 20,
        ignore: float = None
):
    """Generate projections along all axes of an array.

    Parameters
    ----------
    arr : np.ndarray
        Array-like object to project.
    func : Callable
        Numpy-compatible function with which to project along each axis. Must
        accept an "axis" keyword argument.
    step : int, optional
        Step size when chunking `arr`.
        Defaults to 20.
    ignore : float, optional
        Values in `arr` to ignore during computation.
        Defaults to None, in which case all values are used.

    Returns
    -------
    proj : list[np.ndarray]
        Projections along each axis of `arr`.

    Notes
    -----
    Useful for array-like objects (h5py.dataset, zarr.Array, etc.) too large to
    fit in memory.

    Examples
    --------
    Mean intensity projection
    >>> t_arr = np.arange(30).reshape(3, 10)
    >>> t_arr
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
    >>> proj_0, proj_1 = project_arr(t_arr, func=np.mean)
    >>> proj_0
    array([10., 11., 12., 13., 14., 15., 16., 17., 18., 19.])
    >>> proj_1
    array([ 4.5, 14.5, 24.5])

    Projection with zero values ignored
    >>> t_arr[:2, :5] = 0
    >>> t_arr
    array([[ 0,  0,  0,  0,  0,  5,  6,  7,  8,  9],
           [ 0,  0,  0,  0,  0, 15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
    >>> proj_0, proj_1 = project_arr(t_arr, func=np.mean, ignore=0)
    >>> proj_0
    array([20., 21., 22., 23., 24., 15., 16., 17., 18., 19.])
    >>> proj_1
    array([ 7. , 17. , 24.5])
    """
    # if True, compute initial projection via slice-wise comparison
    s_wise = func in (np.min, np.max, np.nanmin, np.nanmax)

    # mask ignore values with np.nan
    if ignore is not None:
        try:
            func = getattr(np, f"nan{func.__name__}")
        except AttributeError:
            pass  # no nan-version exists, use original func

    proj = [
        np.zeros([d for j, d in enumerate(arr.shape) if j != i])
        for i in range(len(arr.shape))]
    dtype = arr[0].dtype

    # chunk along first axis
    for idx in range(0, arr.shape[0], step):
        end = min(idx + step, arr.shape[0])
        sub = arr[idx:end]
        if ignore is not None:
            sub = np.where(sub == ignore, np.nan, sub.astype(float, copy=False))

        # Compute projections for subsequent axes
        for dim in range(1, len(arr.shape)):
            proj[dim][idx:end] = func(sub, axis=dim)

        # Compute slice-wise first dim
        if s_wise:
            val = func(sub, axis=0)
            proj[0] = val if idx == 0 else func(
                np.stack([proj[0], val], axis=0), axis=0)

    # Non-slice-wise first axis
    if not s_wise:
        chunks = []
        for ydx in range(0, arr.shape[1], step):
            sub = arr[:, ydx:min(ydx + step, arr.shape[1])]
            if ignore is not None:
                sub = np.where(
                    sub == ignore, np.nan, sub.astype(float, copy=False))

            chunks.append(func(sub, axis=0))

        proj[0] = np.concatenate(chunks, axis=0)

    for i in range(len(proj)):
        proj[i] = np.nan_to_num(proj[i], nan=ignore) if np.any(
            np.isnan(proj[i])) else proj[i]
        proj[i] = proj[i].astype(dtype) if s_wise else proj[i]

    return proj


def get_mean_std(
        arr: np.ndarray,
        axis: int = 0,
        step: int = 20,
        mean: float = None,
        ignore: float = None
):
    """Apply two-pass algorithm to compute mean and std of large array.

    Parameters
    ----------
    arr : np.ndarray
        Array for which to compute mean and std.
    axis : int, optional
        Axis along which to chunk `dset`.
        Defaults to 0.
    step : int, optional
        Step size when chunking `axis` in `arr`.
        Defaults to 20.
    mean : float, optional
        Mean value in `arr` if already known.
        Defaults to None
    ignore : float, optional
        Values in `arr` to ignore during computation.
        Defaults to None, in which case all values are used.

    Returns
    -------
    mean : float
        Mean value in `arr`.
    std : float
        Standard deviation of values in `arr`.
    """
    idx = [slice(None) for _ in arr.shape]
    idx[axis] = 0
    total_count = 0

    # First pass: mean
    if mean is None:
        total_sum = 0.0
        for i in range(0, arr.shape[axis], step):
            indices = [slice(None) for _ in arr.shape]
            indices[axis] = slice(i, min(i + step, arr.shape[axis]))
            chunk = arr[tuple(indices)]
            chunk = chunk[slice(None) if ignore is None else chunk != ignore]
            total_sum += np.sum(chunk)
            total_count += chunk.size

        mean = total_sum / total_count

    # Second pass: variance
    total_sq_diff = 0.0
    for i in range(0, arr.shape[axis], step):
        indices = [slice(None) for _ in arr.shape]
        indices[axis] = slice(i, min(i + step, arr.shape[axis]))
        chunk = arr[tuple(indices)]
        chunk = chunk[slice(None) if ignore is None else chunk != ignore]
        total_sq_diff += np.sum((chunk - mean) ** 2)

    std = np.sqrt(total_sq_diff / (total_count - 1))
    return mean, std


def arr_to_sparse(
        arr: np.ndarray
):
    """Extract coordinates of nonzero values in an array.

    Parameters
    ----------
    arr : np.ndarray
        Dense array to convert to sparse format.

    Returns
    -------
    coords : np.ndarray
        Coordinates of nonzero values in arr. Has shape (n_nonzero, arr.ndim).
    values : np.ndarray
        Corresponding nonzero values in arr. Has shape (n_nonzero,).

    Examples
    --------
    Extract nonzero values from 3x10 array
    >>> t_arr = np.arange(30).reshape(3, 10)
    >>> t_arr[t_arr % 9 != 0] = 0
    >>> t_arr
    array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  9],
           [ 0,  0,  0,  0,  0,  0,  0,  0, 18,  0],
           [ 0,  0,  0,  0,  0,  0,  0, 27,  0,  0]])
    >>> t_coords, t_values = arr_to_sparse(t_arr)
    >>> t_coords
    array([[0, 9],
           [1, 8],
           [2, 7]])
    >>> t_values
    array([ 9, 18, 27])
    """
    coords = np.nonzero(arr)
    coords, values = np.stack(list(coords), axis=-1), arr[*coords]
    return coords, values


def sparse_to_arr(
        coords: np.ndarray,
        values: np.ndarray,
        dims: list[int] | tuple[int] | np.ndarray[int] = None
):
    """Convert nonzero coordinates and values into a dense array.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of nonzero values in dense array. Shape is
        (nonzero_count, arr.ndim).
    values : np.ndarray
        Corresponding nonzero values in dense array. Shape is (nonzero_count,).
    dims : list[int] | tuple[int] | np.ndarray[int], optional
        Shape of dense array.
        Defaults to None, in which case shape inferred from coordinates.

    Returns
    -------
    dense : np.ndarray
        Dense array. Dtype inferred from values.

    Examples
    --------
    Dense size inferred from coordinates
    >>> t_arr = np.array([[0, 4], [1, 6], [0, 2]])
    >>> t_values = np.array([ 9, 18, 27])
    >>> sparse_to_arr(t_arr, t_values)
    array([[ 0,  0, 27,  0,  9,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 18]])

    Dense size specified
    >>> t_dims = (3, 10)
    >>> sparse_to_arr(t_arr, t_values, t_dims)
    array([[ 0,  0, 27,  0,  9,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 18,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]])
    """
    dims = np.max(coords, axis=0) + 1 if dims is None else dims
    dense = np.zeros(np.asarray(dims, dtype=int), dtype=values.dtype)
    dense[tuple(coords.T.astype(int))] = values
    return dense


def none_concat(
        arr_2: np.ndarray,
        arr_1: np.ndarray = None,
        axis: int = 0
):
    """Concatenate two arrays, with flexible behavior if one array is None.

    Parameters
    ----------
    arr_2 : np.ndarray
        Array to concatenate to the end of a growing sequence.
    arr_1 : np.ndarray, optional
        Array to concatenate to the beginning of a growing sequence. If `arr_1`
        is ``None``, arr_2 is the beginning of the sequence.
        Defaults to None.
    axis : int, optional
        Axis along which to concatenate arrays.
        Defaults to 0.

    Returns
    -------
    arr : np.ndarray
        Concatenated array.

    Examples
    --------
    Concatenate two non-None arrays.
    >>> t_arr_1 = np.arange(5)
    >>> t_arr_2 = np.arange(5) + 5
    >>> none_concat(t_arr_2, t_arr_1)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    Concatenate with None value.
    >>> none_concat(t_arr_2, None)
    array([5, 6, 7, 8, 9])
    """
    arr = arr_2 if arr_1 is None else np.concatenate((arr_1, arr_2), axis=axis)
    return arr


def average_distance(
        arr: np.ndarray
):
    """Compute average pairwise euclidean distance between all points.

    Parameters
    ----------
    arr : np.ndarray
        Has shape (n_points, n_dimensions).

    Returns
    -------
    dist : float
        Average pairwise distance between all points in `arr`.

    Examples
    --------
    >>> test_arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> average_distance(test_arr)
    np.float64(1.4142135623730951)
    """
    dist = pdist(arr, metric="euclidean").mean()
    return dist


def corr_coef(
        arr_1: np.ndarray,
        arr_2: np.ndarray,
        method: str = "p"
):
    """Compute correlations between two sets of vectors.

    Parameters
    ----------
    arr_1 : np.ndarray
        First set of vectors. Has shape (samples_a, observations).
    arr_2 : np.ndarray
        Second set of vectors. Has shape (samples_b, observations).
    method : str, optional
        If s, use Spearman correlation. Otherwise use Pearson correlation.
        Defaults to p.

    Returns
    -------
    corr_matrix : np.ndarray
        Dense correlation matrix with shape (samples_a, samples_b). The value
        at [i, j] is the `method` correlation between the i_th vector in
        `arr_1` and j_th vector in `arr_2`.
    """
    if method == "s":
        arr_1 = np.apply_along_axis(sst.rankdata, axis=1, arr=arr_1)
        arr_2 = np.apply_along_axis(sst.rankdata, axis=1, arr=arr_2)

    arr_1 = arr_1 - arr_1.mean(axis=1, keepdims=True)
    arr_2 = arr_2 - arr_2.mean(axis=1, keepdims=True)

    # Compute numerator: X^T @ Y (dot product of centered columns)
    corr_matrix = arr_1 @ arr_2.T / np.outer(
        np.linalg.norm(arr_1, axis=1), np.linalg.norm(arr_2, axis=1))
    return corr_matrix
