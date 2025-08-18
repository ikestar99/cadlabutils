#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July 4 13:06:21 2023
@author: ike
"""


import numpy as np
import pandas as pd
import scipy.ndimage as scn


"""
Helper functions for modifying imaging data hyperstacks using binary masks.
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


def _mask_zeros(
        mask: np.ndarray
):
    """
    Masks all zero values of an input array using np.ma module.

    Args:
        mask (np.ndarray): Input array.

    Returns:
        Masked array (np.ma.masked_array).

    Raises:
        ValueError: Mask is empty.

    Examples:
        >>> test_mask = np.array([0, 1, 2, 0, 3])
        >>> test_mask = _mask_zeros(test_mask)
        >>> test_mask
        masked_array(data=[--, 1, 1, --, 1],
                     mask=[ True, False, False,  True, False],
                     fill_value=999999)

    Error Examples:
        >>> invalid_mask = np.array([0, 0, 0, 0, 0])
        >>> _mask_zeros(invalid_mask)
        Traceback (most recent call last):
            ...
        ValueError: Mask is empty.
    """
    mask = (mask > 0).astype(int)
    if np.sum(mask) <= 0:
        raise ValueError("Mask is empty.")

    mask = np.ma.masked_where(mask == 0, mask)
    return mask


def label_mask(
        mask: np.ndarray
):
    """
    Labels contiguous regions in a ND mask array and returns an ND + 1 array
    where the first dimension corresponds to the number of regions found, and
    each index along this dimension is a binary integer mask for that region
    with the same shape as the input mask.

    Note:
        label_mask is intended for use with 2D masks corresponding to ground
        truth images, but is written to work with arrays of varying
        dimensionality.

    Args:
        mask (np.ndarray): Integer mask array on which to perform labeling.

    Returns:
        Labeled mask array.

    Example:
        >>> test_mask = np.array([[1, 1, 0], [0, 0, 1]])
        >>> test_array = label_mask(test_mask)
        >>> test_array
        array([[[1, 1, 0],
                [0, 0, 0]],
        <BLANKLINE>
               [[0, 0, 0],
                [0, 0, 1]]])
    """
    labeled_mask, regions = scn.label(mask)
    labeled_mask = np.array([labeled_mask == i for i in range(1, regions + 1)])
    labeled_mask = labeled_mask.astype(int)
    return labeled_mask


def aggregate_region(
        hyperstack: np.ndarray,
        mask: np.ndarray,
        func: np.ufunc = np.nanmean
):
    """
    Measures average value in a region specified by the mask. The hyperstack is
    a 5D array, and the mask is a 2D array of integers where all nonzero
    entries define a region of interest. The func parameter is the numpy
    function that should be used to aggregate values within a region.

    Args:
        hyperstack (np.ndarray): 5D array of images on which to perform
            region of interest aggregation
        mask (np.ndarray): 2D mask of the region within the image from which to
            compute an aggregate. Must have the same shape as the last two
            dimensions of the hyperstack.
        func (np.ufunc): Numpy function to aggregate values within a region.
            Must be a nan-type function to ignore non regions of interest in
            mask. Defaults to np.nanmean.

    Returns:
        Float array with the same shape as the first 3 dimensions on the input
            hyperstack wherein array[t, z, c] corresponds to the func aggregate
            of all values within the region of interest in the image
            hyperstack[t, z, c, :, :].

    Raises:
        ValueError: If the hyperstack and mask shapes do not match.

    Error Examples:
        >>> test_hyperstack = np.random.rand(2, 2, 2, 3, 2)
        >>> test_mask = np.array([[1, 0], [0, 1]])
        >>> aggregate_region(test_hyperstack, test_mask)
        Traceback (most recent call last):
            ...
        ValueError: Hyperstack and mask shapes do not match.
    """
    if hyperstack.shape[-2:] != mask.shape:
        raise ValueError("Hyperstack and mask shapes do not match.")

    mask = _mask_zeros(mask)
    hyperstack = hyperstack * mask
    hyperstack = func(hyperstack, axis=(-2, -1))
    hyperstack = np.array(hyperstack)
    return hyperstack


def subtract_background(
        hyperstack: np.ndarray,
        mask: np.ndarray,
        func: np.ufunc = np.nanmean
):
    """
    For each 2D image in the 5D input hyperstack, subtract the baseline value
    computed using a statistical method specified by the parameter func in the
    background region specified by the mask.

    Args:
        hyperstack (np.ndarray): 5D array of images on which to perform
            background correction.
        mask (np.ndarray): 2D mask of the region within the image from which to
            compute the background.
        func (np.ufunc): Numpy function to aggregate values within background.
            Must be a nan-type function to ignore non regions of interest in
            mask. Defaults to np.nanmean.

    Returns:
        Array of the same shape as the input array with the average background
        pixel intensity of each 2D image subtracted from all other pixels
        within each image.

    Examples:
        >>> test_hyperstack = np.random.rand(2, 1, 1, 2, 2)
        >>> test_mask = np.array([[1, 0], [0, 1]])
        >>> subtract_background(test_hyperstack, test_mask) # random
        array([[[[[-0.25986144, -0.01362817],
                  [-0.16114681,  0.25986144]]]],
        <BLANKLINE>
        <BLANKLINE>
        <BLANKLINE>
               [[[[ 0.3862333 , -0.47352969],
                  [-0.47722033, -0.3862333 ]]]]])
    """
    background = aggregate_region(hyperstack, mask, func)
    background = background[..., np.newaxis, np.newaxis]
    hyperstack = hyperstack - background
    return hyperstack


def project_arr(
        arr: np.ndarray,
        func: np.ufunc,
        step: int = 50
):
    """
    Generate projections along all axes of a z-stack stored in zarr format.

    Args:
        arr (np.ndarray):
            Array to project.
        func (np.ufunc, optional):
            numpy function with which to project along each axis.
            Defaults to np.min.
        step (int, optional):
            Number of z-planes to process at a time. Defaults to 50.

    Returns:
        proj (list[np.ndarray]):
            Contains the following structure:
            -   key (str):
                    Projection axis. "z", "y", "x".
            -   value (np.ndarray):
                    Projection array.
    """
    # project slices along z, y, and x dimensions
    proj = [
        np.zeros([d for j, d in enumerate(arr.shape) if j != i])
        for i in range(len(arr.shape))]
    for zdx in range(0, arr.shape[0], step):
        end = min(zdx + step, arr.shape[0])
        sub = arr[zdx:end]

        # compute projections on subset
        val = func(sub, axis=0)
        proj[0] = val if zdx == 0 else func(
            np.stack([proj[0], val], axis=0), axis=0)
        for dim in range(1, len(arr.shape)):
            proj[dim][zdx:end] = func(sub, axis=dim)

    if func not in (np.min, np.max):
        val = [
            func(arr[:, ydx: min(ydx + step, arr.shape[1])], axis=0)
            for ydx in range(0, arr.shape[1], step)]
        proj[0] = np.concatenate(val, axis=0)

    # return projections
    return proj


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
