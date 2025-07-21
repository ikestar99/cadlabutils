#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import numpy as np
import shutil
import tifffile as tf

from pathlib import Path


"""
File path access and manipulation
===============================================================================
"""


def remove_path(
        path_: Path
):
    """
    Remove a path or directory if it exists.

    Args:
        path_ (Path):
            Path to file or directory to remove.
    """
    if path_.is_dir():
        shutil.rmtree(path_)
    else:
        path_.unlink()


def copy_path(
        src: Path,
        dst: Path
):
    """
    Copy a path to a new location.

    Args:
        src (Path):
            Path to file to copy.
        dst (Path):
            Destination of copied path.
    """
    shutil.copy(src, dst)


def find_first_row(
        path_csv: Path = None,
        raw_text: str = None
):
    """
    Find first numeric row in a csv-like file. Used to skip extraneous comments
    and nonstandard headers when reading data stored in csv format.

    Args:
        path_csv (Path, optional):
            Path to file of interest (swc).
            Defaults to None, in which case raw_text arg cannot be None.
        raw_text (str, optional):
            Entire contents of csv file as string.
            Defaults to None, in which case path_csv arg cannot be None.

    Returns:
        (int):
            Number of non-numeric rows at beginning of file.
    """
    def _check_rows(
            iterable
    ):
        for i, line in enumerate(iterable):
            try:
                # check if the first element is numeric
                float(" ".join(line.strip().split(",")).split(" ")[0])
                return i
            except ValueError:
                # if not numeric, check the next row
                continue

        return None

    # open the swc file
    if path_csv is not None:
        with open(path_csv, 'r') as f:
            skip = _check_rows(f)
    elif raw_text is not None:
        skip = _check_rows(raw_text.splitlines())
    else:
        raise ValueError("Either path_csv or raw_text must be provided")

    return skip


def save_tif(
        array: np.ndarray,
        file: Path,
        dtype: type = np.uint8,
):
    """
    Save array as tif image.

    NOTE: if saving a projection, axes arg can either be the axis order of the
    projection itself ("YX" for a z projection) or the axis projected ("z" or
    "Z").

    Args:
        array (ndarray):
            Array to save.
        file (Path):
            Path to save image data (tif).
        dtype (type, optional):
            Desired datatype of saved array.
            Defaults to np.uint8.
    """
    tf.imwrite(file, array.astype(dtype), compression="zlib")
