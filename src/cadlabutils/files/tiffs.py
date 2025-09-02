#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import numpy as np
import tifffile as tf

from pathlib import Path


def save_tif(
        file: Path,
        arr: np.ndarray,
        axes: str,
        **kwargs
):
    """Save array as ImageJ-compatible tif image with zlib compression.

    Parameters
    ----------
    file : Path
        Path to save image (.tif).
    arr : np.ndarray
        Array to save as image.
    axes : str
        Axis order of tif image. Z stack is "ZYX".

    Other Parameters
    ----------------
    **kwargs : dict
        Keyword arguments passed to tifffile.imwrite.
    """
    tf.imwrite(
        file, arr, imagej=True, metadata={"axes": axes}, compression="zlib",
        **kwargs)
