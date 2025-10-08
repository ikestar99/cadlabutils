#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


# 1. Standard library imports
from pathlib import Path
import re
import shutil


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


def clean_name(
        path: Path,
        fill: str = "_",
        ext: bool = False
):
    """Replace unsafe file name characters with underscores.

    Parameters
    ----------
    path : Path
        Potential file name to clean up.
    fill : str, optional
        Character used to replace unsafe characters in `path`.
        Defaults to "_".
    ext : bool, optional
        If True, preserve original file extension.
        Defaults to False.

    Returns
    -------
    cleaned : Path
        Cleaned up `path` where all characters that are not:
        - alphanumeric
        - " ", "-", "_"
        are replaced with `fill`.

    Examples
    --------
    Clean file name without extension.
    >>> test_path = Path("/this/is/a/test/Th1$ i7 @n +u-n-s@fe n&me*..tif")
    >>> clean_name(test_path)
    PosixPath('/this/is/a/test/Th1_ i7 _n _u-n-s_fe n_me_tif')

    Clean file name with preserved extension.
    >>> clean_name(test_path, ext=True)
    PosixPath('/this/is/a/test/Th1_ i7 _n _u-n-s_fe n_me_.tif')
    """
    cleaned = path.parent.joinpath(
        re.sub(r"[^0-9a-zA-Z-_ ]+", fill, path.stem if ext else path.name))
    cleaned = cleaned.with_suffix(path.suffix) if ext else cleaned
    return cleaned
