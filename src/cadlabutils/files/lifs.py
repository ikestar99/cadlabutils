#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


# 1. Standard library imports
from pathlib import Path

# 2. Third-party library imports
import numpy as np
from readlif.reader import LifFile, LifImage


CHANNEL_ORDER =  ("M", "C", "T", "Z", "Y", "X")


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
    shape : tuple[int]
        Shape of `lif_image` in (M, C, T, Z, Y, X).
    dtype : type
        Data type of `lif_image`.
    """
    shape = (
        lif_image.dims.m, lif_image.channels, lif_image.dims.t,
        lif_image.dims.z, lif_image.dims.y, lif_image.dims.x)
    dtype = np.array(lif_image.get_frame(z=0, t=0, c=0, m=0)).dtype
    return shape, dtype


def get_image(
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


def get_substack(
        lif_image: LifImage,
        c_range: tuple[int, ...] | list[int, ...] | None,
        t_range: tuple[int, ...] | list[int, ...] | None,
        z_range: tuple[int, ...] | list[int, ...] | None,
        m_range: tuple[int, ...] | list[int, ...] | None = (0,)
):
    """Extract image data from Leica Lif Image.

    Parameters
    ----------
    lif_image : LifImage
        Opened lif image.
    c_range : tuple[int, ...] | list[int, ...] | None
        Channel indices to extract. If None, extract all channels.
    t_range : tuple[int, ...] | list[int, ...] | None
        Frame indices to extract. If None, extract all frames.
    z_range : tuple[int, ...] | list[int, ...] | None
        Z slice indices to extract. If None, extract all z slices.

    Returns
    -------
    arr : np.ndarray
        6D array of image data. `arr` has shape:
        (len(`m_range`), len(`c_range`), len(`t_range`), len(`z_range`),
        Y pixels, X pixels).

    Other Parameters
    ----------------
    m_range : tuple[int, ...] | list[int, ...] | None, optional
        Mosaic indices to extract. If None, extract all mosaic indices.
        Defaults to (0,), in which case only first mosaic index is extracted.
    """
    shape, dtype = get_metadata(lif_image)
    new_shape, bounds = [], []
    for d, r in zip(shape[:-2], (m_range, c_range, t_range, z_range)):
        new_shape += [d if r is None else len(r)]
        bounds += [range(d) if r is None else r]

    arr = np.empty(shape=new_shape + list(shape[-2:]), dtype=dtype)
    for mdx, m in enumerate(bounds[0]):
        for cdx, c in enumerate(bounds[1]):
            for tdx, t in enumerate(bounds[2]):
                for zdx, z in enumerate(bounds[3]):
                    arr[mdx, cdx, tdx, zdx] = lif_image.get_frame(
                        z=z, c=c, t=t, m=m)

    return arr


def get_tree(
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


    >> test_tree = get_tree(Path(".../file.lif"))
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


class LifWrapper:
    """Convenience wrapper to index lif images like numpy arrays.

    Attributes
    ----------
    lif_image : LifImage
        Opened lif image.
    dims : tuple[str, ...]
        Non-singleton leading dimensions (all but "Y", "X") present in
        `lif_image`. Subset of ("M", "C", "T", "Z").

    Parameters
    ----------
    lif_image : LifImage
        Opened lif image.

    Notes
    -----
    `LifWrapper` is a convenience wrapper around `get_substack` that provides
    an indexing interface for mosaic tiles, channels, time points, and z slices
    for lif images.
    """
    def __init__(
            self,
            lif_image: LifImage
    ):
        self.lif_image = lif_image
        shape, _ = get_metadata(lif_image)
        self.dims = [
            d for s, d in zip(shape[:-2], CHANNEL_ORDER[:-2]) if s > 1]
        self._crop = [
            slice(None) if d in self.dims else 0 for d in CHANNEL_ORDER[:-2]]

    def __getitem__(
            self,
            *args
    ):
        """Extract 2D "YX" images over a range of leading dimensions

        Parameters
        ----------
        *args
        """
        get_kwargs = {}
        for d, idx in zip(self.dims, args):
            value = [idx] if isinstance(idx, int) else [i for i in idx]
            get_kwargs[f"{d}_range".lower()] = value

        array = get_substack(self.lif_image, **get_kwargs)[*tuple(self._crop)]
        return array
