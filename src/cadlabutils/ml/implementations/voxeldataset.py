#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 09:00:00 2024
@author: ike
"""


# 2. Third-party library imports
import numpy as np
import torch
from torch.utils.data import Dataset

# 3. Local application / relative imports
import cadlabutils.arrays as cdu_a


class VoxelDataset(Dataset):
    def __init__(
            self,
            data,
            voxel: tuple,
            c_o: int,
            pad: tuple = None,
            mask: np.ndarray = None,
            add_channel: bool = True
    ):
        super(VoxelDataset, self).__init__()
        self.shape, self.voxel = np.array(data.shape), np.array(voxel)
        self.pad = np.array(pad or tuple([0] * self.shape.size))
        self.add = add_channel
        if not np.all(self.shape >= voxel):
            raise ValueError(f"Voxel {voxel} must be subset of {data.shape}")

        indices = self.shape // voxel
        self.index = np.arange(np.prod(indices)).reshape(indices)
        self.data = data
        self.mask = mask
        self.output = np.zeros([c_o] + list(self.shape))

    def __len__(
            self
    ):
        return self.index.size

    def __getitem__(
            self,
            idx: int
    ):
        idxs = np.array(np.unravel_index(idx, self.index.shape) * self.voxel)

        start = np.maximum(idxs - self.pad, 0)
        end = np.minimum(idxs + self.voxel + self.pad, self.shape)
        pad_left = -np.minimum(idxs - self.pad, 0)
        pad_right = np.maximum(idxs + self.voxel + self.pad - self.shape, 0)
        raw_s = cdu_a.arr_slice(start, end)
        pad = None if np.sum(pad_left + pad_right) == 0 else (
            tuple((l, r) for l, r in zip(pad_left, pad_right)))

        x = self.data[*raw_s]
        x = x if pad is None else np.pad(x, pad, mode="symmetric")
        x = torch.from_numpy(x[None] if self.add else x)
        if self.mask is None:
            return x

        m = self.mask[*raw_s]
        m = m if pad is None else np.pad(m, pad, mode="symmetric")
        m = torch.from_numpy(m[None] if self.add else m)
        return x, m

    def __setitem__(
            self,
            idx: int,
            value: np.ndarray
    ):
        val_s = cdu_a.arr_slice(self.pad, np.array(value.shape[1:]) - self.pad)
        idxs = np.array(np.unravel_index(idx, self.index.shape) * self.voxel)
        raw_s = cdu_a.arr_slice(idxs, idxs + self.voxel)
        self.output[slice(None), *raw_s] = value[slice(None), *val_s]
