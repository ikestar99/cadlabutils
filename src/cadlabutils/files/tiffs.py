#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


# 1. Standard library imports
from pathlib import Path
import warnings

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
        _shapes = [
            (len(getattr(tif, a)), *getattr(tif, a)[0].shape)
            for a in ("pages", "series")]
        _shapes = [s if s[0] > 1 else s[1:] for s in _shapes]
        shape = max(_shapes, key=lambda x: len(x))
        multipage = len(shape) > 2
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
        Path to a directory of images or to a single multipage image (.tif).
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
    If given a multipage tif with all image data stored in a single page or
    image, will attempt to load full dataset into memory.
    """
    # directory of single .tif images
    if source.is_dir():
        files = natsort.natsorted(source.glob("*.tif"))
        idx = range(len(files)) if i_range is None else i_range
        arr = np.stack([tf.imread(files[i]) for i in idx], axis=0)

    # multipage image
    elif source.is_file and get_metadata(source)[-1]:
        with tf.TiffFile(source) as tif:
            attr = tif.pages if len(tif.pages) > 1 else tif.series
            if len(attr) == 1:
                warnings.warn(
                    f"{source.name} stores tif data in a single block, "
                    f"chunked read is not possible. Will attempt loading "
                    f"stack in its entirety, but doing so may lead to memory "
                    f"errors. To fix this error, store discrete images in "
                    f"tif pages or series, or alternatively extract to a "
                    f"directory of single .tif images.", UserWarning)
                i_range = None
                attr = [getattr(tif, a) for a in ("pages", "series")]
                attr = max(attr, key=lambda x: len(x[0].shape))

            idx = range(len(attr)) if i_range is None else i_range
            arr = attr[0].asarray() if len(idx) <= 1 else np.stack(
                [attr[i].asarray() for i in idx], axis=0)

    return arr


class TiffWrapper:
    """Convenience wrapper to index tif stacks like numpy arrays.

    Attributes
    ----------
    _path : Path

    Parameters
    ----------
    tif_path : Path
        Path to a directory of images or to a single multipage image (.tif).

    Notes
    -----
    `LifWrapper` is a convenience wrapper around `get_substack` that provides
    an indexing interface for a leading dimension (t frames or z slices).
    """

    def __init__(
            self,
            tif_path: Path
    ):
        self._path = tif_path
        self._data = None
        test_path = tif_path if tif_path.is_file() else next(
            tif_path.glob("*.tif*"))
        self.shape, self.dtype, multipage = get_metadata(test_path)
        self.shape = self.shape if multipage else (
            len(list(self._path.glob("*.tif*"))), *self.shape)

    def __getitem__(
            self,
            idx: tuple
    ):
        """Extract 2D "YX" images over a range of leading dimension.

        Parameters
        ----------
        idx : tuple
            Coordinates of image(s) to extract. Indices correspond to
            ("M", "C", "T", "Z").

        Returns
        -------
        np.ndarray
            Extracted image(s). Singleton dimensions in lif file are collapsed,
            whereas singleton dimensions from a single page extraction are
            retained.
        """
        idx = np.atleast_1d(idx).tolist()
        if self._data is None:
            arr = get_substack(
                self._path, np.atleast_1d(np.arange(self.shape[0])[idx]))
            if arr.shape[0] > len(idx):
                self._data = arr
                arr = self._data[idx]
        else:
            arr = self._data[idx]

        return arr
