#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import numpy as np

from pathlib import Path
from readlif.reader import LifFile


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