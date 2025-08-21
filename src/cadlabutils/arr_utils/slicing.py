#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2025
@author: ike
"""


import numpy as np


"""Convenience wrappers around index slicing"""


def arr_slice(
        start: list[int] | tuple[int] | np.ndarray[int],
        stop: list[int] | tuple[int] | np.ndarray[int],
        step: int | list[int] | tuple[int] | np.ndarray[int] = 1,
        end: list[int] | tuple[int] | np.ndarray[int] = None,
        s_buff: int | list[int] | tuple[int] | np.ndarray[int] = 0,
        e_buff: int | list[int] | tuple[int] | np.ndarray[int] = 0
) -> tuple[slice, ...]:
    """
    Generate a tuple of slice objects with set bounds and step size.

    Args:
    ---------------------------------------------------------------------------
        start (list[int] | tuple[int] | np.ndarray[int]):
            Start coordinated of each slice. Length determines number of slice
            objects.
        stop (list[int] | tuple[int] | np.ndarray[int]):
            End coordinated of each slice. Same length as start arg.
        step (int | list[int] | tuple[int] | np.ndarray[int], optional):
            Step size of each slice. If array-like, same length as start arg.
            If integer, use the same step size for all slice dimensions.
            Defaults to 1.
        end (list[int] | tuple[int] | np.ndarray[int], optional):
            Largest valid index for each slice. Same length as start arg.
            Defaults to None, in which case stop values are assumed valid.
        s_buff (int | list[int] | tuple[int] | np.ndarray[int], optional):
            Offset slice start coordinates by set value. If array-like, same
            length as start arg. If integer, use the same offset for all slice
            dimensions. Offset value is subtracted from start value.
            Defaults to 0.
        e_buff(int | list[int] | tuple[int] | np.ndarray[int], optional):
            Offset slice end coordinates by set value. If array-like, same
            length as start arg. If integer, use the same offset for all slice
            dimensions. Offset value is added to end value.
            Defaults to 0.

    Returns:
    ---------------------------------------------------------------------------
        (tuple):
            Slice object for each index in start arg.

    Examples:
    ---------------------------------------------------------------------------
        Generate slices with fixed start and stop coordinates.
        >>> start = [10, 10]
        >>> stop = [20, 20]
        >>> arr_slice(start, stop)
        (slice(10, 20, 1), slice(10, 20, 1))

        Generate slices with flexible start and stop coordinates
        >>> s_buff = [12, 4]
        >>> e_buff = 7
        >>> arr_slice(start, stop, s_buff=s_buff, e_buff=e_buff)
        (slice(0, 27, 1), slice(6, 27, 1))

        Generate slices with flexible coordinates, axis=specific step sizes,
        and known end points
        >>> step = [2, 3]
        >>> end = [25, 25]
        >>> arr_slice(start, stop, step, end, s_buff, e_buff)
        (slice(0, 25, 2), slice(6, 25, 3))
    """
    start = np.array(start) - np.array(s_buff)
    start = np.maximum(start, 0)
    stop = np.array(stop) + np.array(e_buff)
    stop = np.minimum(stop, stop if end is None else end)
    step = np.ones(len(start)) * step if isinstance(step, int) else step
    return tuple(slice(*list(map(int, s))) for s in zip(start, stop, step))
