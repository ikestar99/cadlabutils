#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import os
import time
import numpy as np
import platform
import tracemalloc as tm
import rich.progress as rp

from datetime import datetime


"""
System profiling, i/o management, and aesthetics
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
        desc (str, optional):
            Description of the progress bar.
            Defaults to "".
        tabs (int, optional):
            Number of tabs to offset left edge of progress bar.
            Defaults to 0.

    Returns:
        (rich.progress.ProgressBar):
            Instantiated progress bar.
    """
    return rp.track(item, description=f"{" " * 4 * tabs}{desc}")


def clear_terminal():
    """
    Clear text in current terminal session.
    """
    os.system("cls" if platform.system() == "Windows" else "clear")


def current_date_time(
):
    """
    Get current date and time.

    Returns:
        curr_date (str):
            Current date in YYYY-MM-DD format.
        curr_time (str):
            Current time in hh-mm-ss format.
    """
    curr_date = datetime.now().strftime("%Y-%m-%d")
    curr_time = datetime.now().strftime("%H-%M-%S")
    return curr_date, curr_time


def elapsed_time(
        count: float
):
    """
    Convert elapsed time in seconds to time in hours, minutes, seconds.

    Args:
        count (float):
            Elapsed time in seconds.

    Returns:
        h (int):
            Hour count.
        m (int):
            Remaining minute count.
        s (float):
            Remaining second count.
        (str):
            Elapsed time in hh:mm:ss.ss format.
    """
    h, r = divmod(count, 3600)
    m, s = divmod(r, 60)
    return h, m, s, f"{int(h):02}:{int(m):02}:{s:05.2f}"
