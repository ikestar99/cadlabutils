#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import shutil

from pathlib import Path


def remove_path(
        path: Path
):
    """Remove a path or directory if it exists.

    Parameters
    ----------
    path : Path
        Path to file or directory to remove.
    """
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def copy_path(
        src: Path,
        dst: Path
):
    """Copy a path to a new location.

    Parameters
    ----------
    src : Path
        Path to file to copy.
    dst : Path
        Destination of copied path.
    """
    shutil.copy(src, dst)
