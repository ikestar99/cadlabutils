#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import numpy as np
import scipy.ndimage as scn


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


def round_kernel(
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
        >>> round_kernel(2, radius=2)
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
