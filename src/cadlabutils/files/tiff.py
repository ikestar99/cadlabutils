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
        Path to save image (tif).
    arr : np.ndarray
        Array data to save.
    axes : str
        Axis order of tif image. Z stack is "ZYX". Saved as ImageJ metadata.
    **kwargs
    """
    tf.imwrite(
        file, arr, imagej=True, metadata={"axes": axes}, compression="zlib")
