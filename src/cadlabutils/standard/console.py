#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


# 1. Standard library imports
import datetime
import os
import platform
import time

# 2. Third-party library imports
import psutil
from pympler import asizeof


def clear_terminal():
    """Clear text in current terminal session."""
    os.system("cls" if platform.system() == "Windows" else "clear")


def current_date_time(
):
    """Get current date and time.

    Returns
    -------
    curr_date : str
        Current date in YYYY-MM-DD format.
    curr_time : str
        Current time in hh-mm-ss format.
    """
    curr_date = datetime.now().strftime("%Y-%m-%d")
    curr_time = datetime.now().strftime("%H-%M-%S")
    return curr_date, curr_time


def elapsed_time(
        start: float,
        is_elapsed: bool = False
):
    """Calculate elapsed time in hours, minutes, seconds.

    Parameters
    ----------
    start : float
        Start time in seconds.
    is_elapsed : bool, optional
        If True, interpret `start` as elapsed time.
        Defaults to False, in which case elapsed time is current time -
        `start`.

    Returns
    -------
    h : int
        Hour count.
    m : int
        Remaining minute count.
    s : float
        Remaining second count.
    str
        Elapsed time in hh:mm:ss.ss format.

    Examples
    --------
    Convert start time to elapsed time.
    >>> test_start = time.time()
    >>> time.sleep(1)
    >>> _, _, _, test_str = elapsed_time(test_start)
    >>> test_str
    '00:00:01'

    Convert elapsed time to elapsed time string.
    >>> _, _, _, test_str = elapsed_time(3666, is_elapsed=True)
    >>> test_str
    '01:01:06'
    """
    h, r = divmod(start if is_elapsed else (time.time() - start), 3600)
    m, s = divmod(r, 60)
    return h, m, s, f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def pretty_size(
        memory: int
):
    """Convert memory in bytes to human-readable string.

    Parameters
    ----------
    memory : int
        Memory in bytes.

    Returns
    -------
    str
        Memory as human-readable string.

    Examples
    --------
    10 MiB, 10GiB, 10TiB.
    >>> pretty_size(10 * (1024 ** 2))
    '10.0 MiB'
    >>> pretty_size(10 * (1024 ** 3))
    '10.0 GiB'
    >>> pretty_size(10 * (1024 ** 4))
    '10.0 TiB'
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi"]:
        if abs(memory) < 1024.0:
            return f"{memory:3.1f} {unit}B"
        memory /= 1024.0

    return f"{memory:.1f} Ei{memory}"


def get_ram(
        scale: int = 0
):
    """Report RAM consumption.

    Parameters
    ----------
    scale : int, optional
        Scale memory in bytes to a power of 2^10.
        Defaults to 0, in which case returned memories are in bytes.

    Returns
    -------
    allocated : int
        Active memory pool.
    total : int
        Total memory pool.
    """
    scalar = 1024 ** scale
    vm = psutil.virtual_memory()
    allocated = (vm.total - vm.available) / scalar
    total = vm.total / scalar
    return allocated, total


def get_memory_repeat(
        obj: object,
        scalar: float = 0.75
):
    """Get number of objects that can safely fit in available system memory.

    Parameters
    ----------
    obj : object
        Object to tile in available system memory.
    scalar : float, optional
        Fraction of peak repeats to return as optimum count.
        Defaults to 0.75.

    Returns
    -------
    repeats : int
        Number of `obj` copies that can safely fit in available system memory.
    """
    obj_size = asizeof.asizeof(obj)
    ram_used, ram_tot = get_ram()
    repeats = int(((ram_tot - ram_used) // obj_size) * scalar)
    repeats = max(1, repeats)
    return repeats
