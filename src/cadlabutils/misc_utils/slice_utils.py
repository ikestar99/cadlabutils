#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import numpy as np


"""
Helper functions for manipulating slice objects
===============================================================================
"""


def bound_slice(
        start: list | tuple | np.ndarray,
        end: list | tuple | np.ndarray,
        step: list | tuple | np.ndarray = None
):
    """
    Generate a sequence of slice objects with set bounds and step size.

    Args:
        start (list | tuple | np.ndarray):
            Start coordinated of each slice. Length determines number of slice
            objects.
        end (list | tuple | np.ndarray):
            End coordinated of each slice. Same length as start arg.
        step (list | tuple | np.ndarray, optional):
            Step size of each slice. Same length as start arg.
            Defaults to None, in which case all slices have a step size of 1.

    Returns:
        slices (tuple):
            Slices.
    """
    step = np.ones(len(start)) if step is None else step
    slices = tuple(slice(*list(map(round, s))) for s in zip(start, end, step))
    return slices
