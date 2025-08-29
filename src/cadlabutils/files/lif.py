#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import numpy as np

from pathlib import Path
from readlif.reader import LifFile


def lif_tree(
        file: Path
):
    """Extract structure of a Leica Image File from available metadata.

    Parameters
    ----------
    file : Path
        Path to file (.lif).

    Returns
    -------
    tree : dict[str: dict[str, int | tuple]]
        Structure of stored data. Keys are names of jobs stored in file. Value
        is another dictionary with the following structure per job:
        -   "T Z Y X": (time points, z planes, y planes, x planes).
        -   "Hz µm/p": resolutions in (Hz, z µm/pixel, y µm/pixel, x µm/pixel)
        -   "channel": number of channels stored in image
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
    └── 2024.02.13.MTG.S1.C1.C2.Z
        ├── T Z Y X: (1, 147, 5797, 3921)
        ├── Hz µm/p: ('-', 0.41855, 0.10317, 0.10317)
        ├── channel: 1
        └── mosaic: []
    """
    tree = {}
    for image in LifFile(file).get_iter_image():
        d = image.dims

        # reorder resolution values to T, Z, Y, X
        r = ["-" if r is None else r for r in list(image.scale[::-1])]

        # convert spatial resolutions to ¨m/pixel
        r[1:] = [r if isinstance(r, str) else 1/r for r in r[1:]]
        tree[image.name] = {
            "T Z Y X": (d.t, d.z, d.y, d.x),
            "Hz µm/p": tuple(r),
            "channel": image.channels,
            "mosaic": image.mosaic_position}

    return tree


def unpack_lif(
        sample_dir: Path,
        data_h5: Path,
        data_str: str,
        voxel: tuple,
        invert: bool,
        lif_idx: int = 0,
        **kwargs
):
    """
    Unpack Leica.lif file into directory of single tif images. File must be
    within the directory tree of the current sample.

    Args:
        sample_dir (Path):
            Directory containing sample of interest.
        lif_idx (int, optional):
            Position of image of interest within lif file.
            Defaults to 0, in which case image of interest is the first and/or
            only image stored.
        **kwargs (dict):
            Keyword arguments.
    """
    # skip if data unavailable or sample already processed
    template = "raw {}.tif"
    lif_file = list(sample_dir.rglob("*.lif"))
    if len(lif_file) == 0 or data_h5.is_file():
        return "skipped"

    # open data in .lif file
    reader = LifFile(lif_file[0]).get_image(lif_idx)

    # calculate data shape
    old_shape = (reader.dims.z, *np.array(reader.get_frame(z=0)).shape)
    new_shape = tuple(o - (o % v) for o, v in zip(old_shape, voxel))
    start = tuple((o - n) // 2 for o, n in zip(old_shape, new_shape))
    crop = tuple(slice(s, s + n) for s, n in zip(start, new_shape))

    # read data in .lif and transfer to .h5
    holder = np.zeros((voxel[0], *new_shape[1:]), dtype=np.uint8)
    with h5.File(data_h5, "w") as data:
        # generate dataset for downloaded data
        dset = make_h5_dataset(data, name=data_str, shape=new_shape)

        # transfer data from .lif to .h5
        for z in pbar(range(0, new_shape[0], voxel[0]), desc="unpack"):
            delta = min(voxel[0], new_shape[0] - z)
            for j in range(delta):
                holder[j] = np.array(
                    reader.get_frame(z=z+j+start[0]))[*crop[1:]]

            dset[z:z+delta] = (
                255 - holder[:delta] if invert else holder[:delta])

    for k, v in project_h5(data_h5, data_str, np.max).items():
        save_tif(v, sample_dir.joinpath(template.format(k)), "YX")
