#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2025
@author: ike
"""


# 1. Standard library imports
from typing import Union

# 2. Third-party library imports
import numpy as np


IntArrayLike = Union[list[int] | tuple[int] | np.ndarray[int]]


def arr_slice(
        start: IntArrayLike,
        stop: IntArrayLike = None,
        step: int | IntArrayLike = 1,
        end: IntArrayLike = None,
        s_buff: int | IntArrayLike = 0,
        e_buff: int | IntArrayLike = 0
):
    """Generate multiple slice objects with flexible behavior at bounds.

    Parameters
    ----------
    start : IntArrayLike
        Start coordinated of each slice. Length determines number of slice
        objects.
    stop : IntArrayLike, optional
        End coordinated of each slice. Same length as `start`.
        Defaults to None, in which case `start` is used.
    step : int | IntArrayLike, optional
        Step size of each slice. If array-like, same length as `start`. If
        integer, use the same step size for all slice dimensions.
        Defaults to 1.
    end : IntArrayLike, optional
        Largest valid index for each slice. Same length as `start`.
        Defaults to None, in which case stop values are assumed valid.
    s_buff : int | IntArrayLike, optional
        Offset slice start coordinates by set value. If array-like, same length
        as `start`. If integer, use the same offset for all slice dimensions.
        Offset value is subtracted from start value.
        Defaults to 0.
    e_buff : IntArrayLike, optional
        Offset slice end coordinates by set value. If array-like, same
        length as `start`. If integer, use the same offset for all slice
        dimensions. Offset value is added to end value.
        Defaults to 0.

    Returns
    -------
    tuple
        Slice object for each index in `start`.

    Examples
    --------
    Generate slices with fixed start and stop coordinates.
    >>> t_start = [10, 10]
    >>> t_stop = [20, 20]
    >>> arr_slice(t_start, t_stop)
    (slice(10, 20, 1), slice(10, 20, 1))

    Generate slices with flexible start and stop coordinates
    >>> t_s = [12, 4]
    >>> t_e = 7
    >>> arr_slice(t_start, t_stop, s_buff=t_s, e_buff=t_e)
    (slice(0, 27, 1), slice(6, 27, 1))

    Generate slices with flexible coordinates, axis=specific step sizes,
    and known end points
    >>> t_step = [2, 3]
    >>> t_end = [25, 25]
    >>> arr_slice(t_start, t_stop, t_step, t_end, s_buff=t_s, e_buff=t_e)
    (slice(0, 25, 2), slice(6, 25, 3))
    """
    stop = [s for s in start] if stop is None else stop
    start = np.array(start) - np.array(s_buff)
    start = np.maximum(start, 0)
    stop = np.array(stop) + np.array(e_buff)
    stop = stop if end is None else np.minimum(stop, end)
    step = np.ones(len(start)) * step if isinstance(step, int) else step
    return tuple(slice(*list(map(int, s))) for s in zip(start, stop, step))


def dim_start_end(
        start: int,
        width: int,
        step: int,
        pad: int,
        factor: int,
):
    if width < np.ceil((step + (2 * pad)) / factor) * factor:
        raise ValueError(
            f"Width {width} cannot accommodate step {step} with pad {pad}")

    x0 = max(0, start - pad)
    x1 = min(width, start + step + pad)
    if x0 == 0 and (x1 - x0) % factor != 0:
        x1 += factor - ((x1 - x0) % factor)

    if x1 == width and (x1 - x0) % factor != 0:
        x0 -= factor - ((x1 - x0) % factor)

    return x0, x1


if __name__ == "__main__":
    print(dim_start_end(384, 512, 384, 64, 64))