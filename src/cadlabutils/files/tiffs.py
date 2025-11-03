#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


# 1. Standard library imports
from pathlib import Path

# 2. Third-party library imports
import natsort
import numpy as np
import tifffile as tf


def get_metadata(
        file: Path
):
    """Extract shape and data type of tif image.

    Parameters
    ----------
    file : Path
        Path to image (.tif).

    Returns
    -------
    shape : tuple[int]
        Shape of `file` image in (*P, Y, X, *C).
        -   P is included iff `file` is a multipage image. Corresponds to
            number of pages stored.
        -   C is included iff `file` includes a color channel. Corresponds to
            number of channels stored.
    dtype : type
        Data type of first page in `file`.
    multipage : bool
        True if `file` is a multipage image.
    """
    with tf.TiffFile(file) as tif:
        multipage = len(tif.pages) > 1
        shape = tif.pages[0].shape
        shape = tuple([len(tif.pages)] + list(shape)) if multipage else shape
        dtype = tif.pages[0].dtype

    return shape, dtype, multipage


def get_substack(
        source: Path,
        i_range: tuple[int, ...] | list[int, ...] | None
):
    """
    Iterate over TIFF images as 3D stacks (Z, Y, X, *C).

    Parameters
    ----------
    source : str or Path
        Path to a directory of images of to a single multipage image (.tif).
    i_range : tuple[int, ...] | list[int, ...] | None, optional
        Image indices to extract. If None, extract all images.

    Returns
    -------
    arr : np.ndarray
        3D array of image data. `arr` has shape:
        (len(`i_range`), Y pixels, X pixels, *Channels).

    Notes
    -----
    Individual tif pages are sorted in natural order by file name prior to
    indexing. `i_range` indices should refer to this 0-indexed order.
    """
    # directory of single .tif images
    if source.is_dir():
        files = sorted(source.glob("*.tif"))
        idx = range(len(files)) if i_range is None else i_range
        arr = np.stack([tf.imread(files[i]) for i in idx], axis=0)

    # multipage image
    elif source.is_file and ".tif" in source.name:
        with tf.TiffFile(source) as tif:
            idx = range(len(tif.pages)) if i_range is None else i_range
            arr = np.stack([tif.pages[i].asarray() for i in idx], axis=0)

    return arr
