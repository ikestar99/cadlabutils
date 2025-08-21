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


CIRCLE_FORMULAS = {
    "circumference": lambda x: 2 * np.pi * x,
    "area": lambda x: np.pi * (x ** 2),
    "surface area": lambda x: 4 * np.pi * (x ** 2),
    "volume": lambda x: (4 / 3) * np.pi * (x ** 3),
}


def radial_measure(
        radius: float,
        mode: str,
        as_int: bool = False
):
    """
    Compute geometric measure of circle or sphere with specified radius.

    Args:
        radius (float):
            Radius of circle.
        mode (str):
            Measure to calculate. Valid options are:
            - circumference
            - area
            - surface area
            - volume
        as_int (bool, optional):
            If True, round measure to nearest integer.
            Defaults to False.

    Returns:
        measure (int):
            Measure of circle or sphere with specified radius.

    Examples:
    ---------------------------------------------------------------------------
        Compute circumference, area, surface area, and volume of radius 1.
        >>> radius = 1
        >>> radial_measure(radius, mode="circumference")
        6.283185307179586
        >>> radial_measure(radius, mode="area")
        3.141592653589793
        >>> radial_measure(radius, mode="surface area")
        12.566370614359172
        >>> radial_measure(radius, mode="volume")
        4.1887902047863905
        >>> radial_measure(radius, mode="volume", as_int=True)
        4
    """
    if mode not in CIRCLE_FORMULAS:
        raise ValueError(f"Mode {mode} not recognized")

    measure = CIRCLE_FORMULAS[mode](radius)
    measure = int(round(measure)) if as_int else measure
    return measure


def get_round(
        ndim: int,
        radius: int
):
    """
    Generate a spherical structuring element for an array of arbitrary
    dimensionality.

    Args:
    ---------------------------------------------------------------------------
        ndim (int):
            Number of dimensions in array of interest.
        radius (int):
            Radius of spherical structure.

    Returns:
    ---------------------------------------------------------------------------
        (np.ndarray):
            Spherical structuring element.

    Examples:
    ---------------------------------------------------------------------------
        Generate 2d structuring element witha  radius of 4 pixels.
        >>> get_round(2, radius=2)
        array([[0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1],
               [0, 1, 1, 1, 0],
               [0, 0, 1, 0, 0]])
    """
    struct = np.ones([2 * radius + 1] * ndim)
    struct[tuple([radius] * ndim)] = 0
    struct = (scn.distance_transform_edt(struct) <= radius).astype(int)
    return struct.astype(int)


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
        >>> mask.astype(int)
        array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
               [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]])
    """
    connect = arr.ndim if connect is None else min(connect, arr.ndim)
    arr, _, _ = label_array(arr, connect)
    arr = arr if np.unique(arr).size <= 2 else skm.remove_small_objects(
        arr, min_size, connect)
    return arr != 0
