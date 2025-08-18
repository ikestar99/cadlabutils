#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import numpy as np
import scipy.stats as sst
import scipy.ndimage as scn
import skimage.morphology as skm


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
