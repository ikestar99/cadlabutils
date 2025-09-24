#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import os
import time
import platform

from datetime import datetime


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
        start: float
):
    """Calculate elapsed time in hours, minutes, seconds.

    Parameters
    ----------
    start : float
        Start time in seconds.

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
    Convert seconds to elapsed time.
    >>> test_start = time.time()
    >>> time.sleep(1)
    >>> _, _, _, test_str = elapsed_time(test_start)
    >>> test_str
    '00:00:01.01'
    """
    h, r = divmod(time.time() - start, 3600)
    m, s = divmod(r, 60)
    return h, m, s, f"{int(h):02}:{int(m):02}:{s:05.2f}"


def pretty_size(
        bytes: int
):
    """Convert memory in bytes to human-readable string.

    Parameters
    ----------
    bytes : int
        Memory in bytes.

    Returns
    -------
    str
        Memory as human-readable string.
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi"]:
        if abs(bytes) < 1024.0:
            return f"{bytes:3.1f} {unit}B"
        bytes /= 1024.0

    return f"{bytes:.1f} Ei{bytes}"
