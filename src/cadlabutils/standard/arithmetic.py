#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 09:00:00 2025
@author: Ike
"""


# 2. Third-party library imports
import numpy as np


CIRCLE_FORMULAS = {
    "c": lambda x: 2 * np.pi * x,
    "a": lambda x: np.pi * (x ** 2),
    "s": lambda x: 4 * np.pi * (x ** 2),
    "v": lambda x: (4 / 3) * np.pi * (x ** 3),
}


def radial_measure(
        radius: float,
        mode: str,
        as_int: bool = False
):
    """Compute geometric measure of circle or sphere with specified radius.

    Parameters
    ----------
    radius : float
        Radius of circle.
    mode : str
        Measure to calculate. Valid options are:
        - "c": circumference
        - "a": area
        - "s": surface area
        - "v": volume
    as_int : bool, optional
        If True, round measure to nearest integer.
        Defaults to False.

    Returns
    -------
    measure : int
        Measure of circle or sphere with specified radius.

    Examples
    --------
    Compute circumference, area, surface area, and volume of radius 1.
    >>> radius = 1
    >>> radial_measure(radius, mode="c")
    6.283185307179586
    >>> radial_measure(radius, mode="a")
    3.141592653589793
    >>> radial_measure(radius, mode="s")
    12.566370614359172
    >>> radial_measure(radius, mode="v")
    4.1887902047863905
    >>> radial_measure(radius, mode="v", as_int=True)
    4
    """
    if mode not in CIRCLE_FORMULAS:
        raise ValueError(f"Mode {mode} not recognized")

    measure = CIRCLE_FORMULAS[mode](radius)
    measure = int(round(measure)) if as_int else measure
    return measure


def find_closest_multiples(
        x: float,
        scalar: float
):
    """Find nearest multiples of specified scalar around an input number.

    Parameters
    ----------
    x : float
        Input value around which to find multiples.
    scalar : float
        Find values around `x` that are multiples of `scalar`.

    Returns
    -------
    float
        First multiple of `scalar` <= `x`.
    float
        Second multiple of `scalar` >= `x`.

    Examples
    --------
    `x` is not divisible by `scalar`.
    >>> find_closest_multiples(40, scalar=16)
    (32, 48)

    `x` is divisible by `scalar.
    >>> find_closest_multiples(48, scalar=16)
    (48, 48)

    `x` is smaller than `scalar`.
    >>> find_closest_multiples(15, scalar=16)
    (0, 16)
    """
    factor, rem = divmod(x, scalar)
    return factor * scalar, (factor + int(rem > 0)) * scalar
