#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


# 2. Third-party library imports
import numpy as np
import scipy.ndimage as scn


def round_kernel(
        ndim: int,
        radius: int
):
    """Generate a spherical structuring element.

    Parameters
    ----------
    ndim : int
        Number of dimensions in array of interest.
    radius : int
        Radius of spherical structure.

    Returns
    -------
    struct : np.ndarray
        Spherical structuring element.

    Examples
    --------
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
