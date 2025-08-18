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
        array: np.ndarray,
        file: Path,
        axes: str,
        dtype: type = np.uint8,
):
    """
    Save array as tif image.

    Args:
        array (ndarray):
            Array to save as tif.
        file (Path):
            Path to tif save location.
        axes (str):
            Axis order of tif image or hyperstack. 2D image is "YX", z stack is
            "ZYX". Saved as tif ImageJ metadata.
        dtype (type, optional):
            Desired datatype of saved array.
            Defaults to np.uint8.
    """
    tf.imwrite(
        file, array.astype(dtype), imagej=True, metadata={"axes": axes},
        compression="zlib")
