#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import numpy as np

from pathlib import Path
from readlif.reader import LifFile, LifImage


def get_metadata(
        lif_image: LifImage
):
    """Extract shape and data type of lif image.

    Parameters
    ----------
    lif_image : LifImage
        Opened lif image.

    Returns
    -------
    shape : tuple
        Shape of `lif_image` in (M, C, T, Z, Y, X).
    dtype : type
        Data type of `lif_image`.
    """
    shape = (
        lif_image.dims.m, lif_image.channels, lif_image.dims.t,
        lif_image.dims.z, lif_image.dims.y, lif_image.dims.x)
    dtype = np.array(lif_image.get_frame(z=0, t=0, c=0, m=0)).dtype
    return shape, dtype


def get_lif_image(
        file: Path,
        idx: int
):
    """Extract image from Leica Image File.

    Parameters
    ----------
    file : Path
        Path to Leica Image File (.lif).
    idx : int
        Index of image to extract.

    Returns
    -------
    lif_image : LifImage
        Opened lif image.
    """
    lif_image = LifFile(file).get_image(idx)
    return lif_image


def get_image_substack(
        lif_image: LifImage,
        c_range: tuple[int, ...] | slice | None,
        t_range: tuple[int, ...] | slice | None,
        z_range: tuple[int, ...] | slice | None,
        m_range: tuple[int, ...] | slice | None = (0,)
):
    """Extract image data from Leica Lif Image.

    Parameters
    ----------
    lif_image : LifImage
        Opened lif image.
    c_range : tuple[int, ...] | slice | None
        Channel indices to extract. If None, extract all channels.
    t_range : tuple[int, int] or None
        Frame indices to extract. If None, extract all time points.
    z_range : tuple[int, int] or None
        Z slice indices to extract. If None, extract all z slices.

    Returns
    -------
    np.ndarray
        6D array of image data. `arr` has shape:
        (len(`m_range`), len(`c_range`), len(`t_range`), len(`z_range`),
        Y pixels, X pixels).

    Other Parameters
    ----------------
    m_range : tuple[int, ...] | slice | None
        Mosaic indices to extract. If None, extract all mosaic indices.
        Defaults to (0,), in which case only first mosaic index is extracted.
    """
    (max_m, max_c, max_t, max_z, _, _), _ = get_metadata(lif_image)
    arr = []
    for m in (slice(max_m) if m_range is None else m_range):
        c_arr = []
        for c in (slice(max_c) if c_range is None else c_range):
            t_arr = []
            for t in (slice(max_t) if t_range is None else t_range):
                z_arr = []
                for z in (slice(max_z) if z_range is None else z_range):
                    z_arr += [lif_image.get_frame(z=z, c=c, t=t, m=m)]

                t_arr += [np.stack(z_arr, axis=0)]
            c_arr += [np.stack(t_arr, axis=0)]
        arr += [np.stack(c_arr, axis=0)]
    return np.stack(arr, axis=0)


def lif_tree(
        file: Path
):
    """Extract structure of a Leica Image File from available metadata.

    Parameters
    ----------
    file : Path
        Path to Leica Image File (.lif).

    Returns
    -------
    tree : dict[str: dict[str, int | tuple]]
        Structure of stored data. Keys are "index: name" of jobs in file. Value
        is another dictionary with the following structure per job:
        -   "dtype": data type of stored pixel values.
        -   "T Z Y X": (time points, z planes, y planes, x planes).
        -   "Hz µm/p": resolutions in (Hz, z µm/pixel, y µm/pixel, x µm/pixel).
        -   "channel": number of channels stored in image.
        -   "mosaic": position of mosaic tile if used.

    Notes
    -----
    - See https://readlif.readthedocs.io/en/stable/ for additional information
      on mosaic tiling using readlif.

    Examples
    --------
    >> from ... import print_tree


    >> test_tree = lif_tree(Path(".../file.lif"))
    >> print_tree(test_tree, color=False)
    2024.02.13.MTG.S1.C1.C2.lif
    └── 0: 2024.02.13.MTG.S1.C1.C2.Z
        ├── dtype: uint8
        ├── channel: 1
        ├── T Z Y X: (1, 147, 5797, 3921)
        ├── Hz µm/p: ('-', 0.41855, 0.10317, 0.10317)
        └── mosaic: []
    """
    tree = {}
    for i, image in enumerate(LifFile(file).get_iter_image()):
        dims, dtype = get_metadata(image)

        # reorder resolution values to T, Z, Y, X
        r = ["-" if r is None else r for r in list(image.scale[::-1])]

        # convert spatial resolutions to ¨m/pixel
        r[1:] = [r if isinstance(r, str) else 1/r for r in r[1:]]
        tree[f"{i}: {image.name}"] = {
            "dtype": dtype,
            "channel": dims[1],
            "T Z Y X": dims[2:],
            "Hz µm/p": tuple(r),
            "mosaic": image.mosaic_position}

    return tree
