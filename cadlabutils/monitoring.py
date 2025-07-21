#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import time
import numpy as np
import tracemalloc as tm
import rich.progress as rp


UNITS = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"]


"""
System memory/runtime profiling and aesthetics
===============================================================================
"""


def pbar(
        item,
        desc: str = "",
        tabs: int = 0
):
    """
    Generate a colorful progress bar in the terminal.

    Args:
        item (iterable):
            Iterable to wrap in progress bar.
        desc (str):
            Description of the progress bar.
            Defaults to "".
        tabs (int):
            Number of tabs to offset left edge of progress bar.
            Defaults to 0.
    """
    return rp.track(list(item), description=f"{" " * 4 * tabs}{desc}")


def sec_to_hour_min_sec(
        count: float
):
    """
    Convert elapsed time in seconds to time in hours, minutes, seconds.

    Args:
        count (float):
            Elapsed time in seconds.

    Returns:
        (tuple):
            Contains the following 3 elements:
            -   h (int):
                    Hour count.
            -   m (int):
                    Minute count.
            -   s (float):
                    Second count.
    """
    h, r = divmod(count, 3600)
    m, s = divmod(r, 60)
    return h, m, s


def byte_to_str(
        count: float
):
    """
    Convert memory in bytes to largest applicable abbreviation.

    Args:
        count (float):
            Memory in bytes.

    Returns:
        (str):
            Memory in readable bibyte format, ex 12345678 --> 11.77 MiB.
    """
    idx = 0
    while count >= 1024:
        idx += 1
        count /= 1024

    return f"{count:.2f} {UNITS[min(idx, len(UNITS) - 1)]}"


def profile_func(
        func,
        **kwargs
):
    """
    Profile the runtime and memory consumption of a function.

    Args:
        func (function):
            Function to be profiled.
        kwargs (dict):
            Keyword arguments passed to function.

    Returns:
        (tuple):
            Contains the following two elements:
            -   0 (np.ndarray):
                    1d array with the following 6 elements:
                    -   0 (str):
                            Name of profiled function.
                    -   1 (int):
                            Execution time in hours.
                    -   2 (int):
                            Residual execution time in minutes.
                    -   3 (float):
                            Residual execution time in seconds.
                    -   4 (str):
                            Instantaneous memory consumption after execution.
                            Suffixes are based on base 2.
                    -   5 (str):
                            Peak memory consumption during execution. Suffixes
                            are based on base 2.
            -   1 (varies):
                    Output of function(**kwargs).
    """
    tm.start()
    t0 = time.time()
    output = func(**kwargs)
    h, m, s = sec_to_hour_min_sec(time.time() - t0)
    current, peak = tm.get_traced_memory()
    current, peak = byte_to_str(current), byte_to_str(peak)
    tm.stop()
    return np.array([func.__name__, h, m, s, current, peak]), output
