#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July  4 13:06:21 2023
@author: ike
"""

import numpy as np
import scipy.ndimage as scn
import skimage.measure as skm


"""
Helper functions for modifying imaging data hyperstacks using binary masks.
"""


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
