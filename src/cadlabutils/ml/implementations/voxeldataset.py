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
    """Simple dataset to extract subvoxels from image volumes.

    Attributes
    ----------
    data : np.ndarray
        Image data from which voxels are extracted.
    mask : np.ndarray
        Mask data. Pixelwise correspondence to `data`.
    shape : np.ndarray
        Shape of `data`.
    voxel: np.ndarray
        Shape of sub-voxels to extract from `data`.
    pad: np.ndarray
        Symmetric padding added to each dimension of extracted sub-voxels.
        Effective shape of each sub-voxels is `voxel` + 2(`pad`)
    undo_pad : tuple[slice, ...]
        Tuple of slices used to undo padding from extracted sub-voxels. Has
        one index per dimension in `data`.
    add : bool
        If True, add a singleton channel dimension to extracted sub-voxels.
    _index : np.ndarray
        Tracks valid linear index of each sub-voxel

    Parameters
    ----------
    data
        Image data to convert into voxels. Can be any array-like that supports
        multidimensional slice indexing and defines a shape attribute.
    voxel : tuple | list | np.ndarray
        Shape of sub-voxels to extract from `data`.
    pad : tuple | list | np.ndarray, optional
        Symmetric padding added to each dimension of extracted sub-voxels.
    mask : np.ndarray, optional
        Pixelwise targets for `data`.
        Defaults to None.
    add_channel : bool, optional
        If True, add a singleton channel dimension to extracted sub-voxels.
        Defaults to False.
    drop_empty : bool, optional
        If True, drop empty (uniform value) sub-voxels from extracted
        sub-voxels.
        Defaults to False.

    Raises
    ------
    ValueError
        Instantiating with a voxel size larger than the underlying image data.
    """
    def __init__(
            self,
            data,
            voxel: tuple | list | np.ndarray,
            pad: tuple | list | np.ndarray = None,
            mask: np.ndarray = None,
            add_channel: bool = False,
            drop_empty: bool = False
    ):
        super(VoxelDataset, self).__init__()
        self.data, self.mask = data, mask
        self.shape, self.voxel = np.array(data.shape), np.array(voxel)
        if not np.all(self.shape >= voxel):
            raise ValueError(f"Voxel {voxel} must be subset of {data.shape}")

        self.add = add_channel
        self.pad = np.array(pad) if pad is not None else np.zeros(len(voxel))
        self.undo_pad = cdu_a.arr_slice(self.pad, self.pad + self.voxel)
        self._grid = self.shape // voxel
        self._index = np.arange(np.prod(self._grid))
        if drop_empty:
            for i in range(self._index.size):
                x = self.data[self._idx_to_slice(i)[1]]
                self._index[i] = -1 if x.min() == x.max() else i

            self._index = self._index[self._index >= 0]

    def __len__(
            self
    ):
        """Get length of instance.

        Returns
        -------
        int
            Number of sub-voxels stored.
        """
        return self._index.size

    def __getitem__(
            self,
            idx: int
    ):
        """Get sub-voxel at specified index.

        Parameters
        ----------
        idx : int
            Index of sub-voxel to pull.

        Returns
        -------
        x : torch.tensor
            Sub-voxel as a tensor.
        m : torch.tensor
            Corresponding mask for `x` as a tensor. Only returned if instance
            `mask` attribute is not None.
        """
        raw_s, vox_s, pad = self._idx_to_slice(idx)

        x = self.data[*raw_s]
        x = x if pad is None else np.pad(x, pad, mode="reflect")
        x = torch.from_numpy(x[None] if self.add else x)
        if self.mask is not None:
            m = self.mask[*raw_s]
            m = m if pad is None else np.pad(m, pad, mode="reflect")
            m = torch.from_numpy(m[None] if self.add else m)
            return x, m

        return x

    def _idx_to_slice(
            self,
            idx
    ):
        """Get slice indices of sub-voxel at specified index.

        Parameters
        ----------
        idx : int
            Flat index of sub-voxel to pull.

        Returns
        -------
        raw_s : tuple[slice, ...]
            Slice indices of `idx`th sample along every axis of instance `data`
            attribute.
        vox_s : torch.tensor
            Slice indices of `idx`th sample along every axis of instance `data`
            attribute, prior to any padding. In the absence of padding, `raw_s`
            and `vox_s` are equivalent.
        pad : tuple[tuple[int, int], ...] | None
            At boundary cases, additional padding needed per axis. None if no
            additional padding is needed.
        """
        idxs = np.array(
            np.unravel_index(self._index[idx], self._grid) * self.voxel)

        start = np.maximum(idxs - self.pad, 0)
        end = np.minimum(idxs + self.voxel + self.pad, self.shape)
        pad_left = -np.minimum(idxs - self.pad, 0)
        pad_right = np.maximum(idxs + self.voxel + self.pad - self.shape,
                               0)
        raw_s = cdu_a.arr_slice(start, end)
        vox_s = cdu_a.arr_slice(
            idxs, np.minimum(idxs + self.voxel, self.shape))
        pad = None if np.sum(pad_left + pad_right) == 0 else (
            tuple((l, r) for l, r in zip(pad_left, pad_right)))
        return raw_s, vox_s, pad

    def unpadded_slice(
            self,
            idx
    ):
        return self._idx_to_slice(idx)[1]
