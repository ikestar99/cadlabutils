#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4 13:06:21 2024
@author: ike
"""


import numpy as np
import pandas as pd

from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedKFold


class CoreDataset(Dataset):
    """
    CoreDataset object stores a mapping between a set of hierarchically ordered
    metadata values and the index of a corresponding datum.

    Class Attributes:
    ---------------------------------------------------------------------------
        _INDEX (str):
            Internal metadata level to track corresponding index of each datum.

    Instance Attributes:
    ---------------------------------------------------------------------------
        meta (pd.DataFrame):
            Hierarchically indexed DataFrame storing metadata.

    Example:
    ---------------------------------------------------------------------------
        Instantiate with 1d datum.
        >>> # arbitrary dataset with 4 samples
        >>> length = 4
        >>> # arbitrary metadata values, including constants and lists
        >>> day, label, count = "Mon", ["cortex", "thalamus"], [3, 6, 1, 4]
        >>> dataset = CoreDataset(length, day=day, label=label, count=count)
        >>> dataset.meta  # doctest: +NORMALIZE_WHITESPACE
                            _data_index
        day label    count
        Mon cortex   1                2
                     3                0
            thalamus 4                3
                     6                1
    """
    _INDEX = "_data_index"

    def __init__(
            self,
            n_samples: int,
            **kwargs
    ):
        """
        Args:
            n_samples (int):
                Number of samples or observations included in dataset.
            **kwargs (int):
                Metadata associated with each sample in the dataset. The order
                of keyword arguments dictates the hierarchical order of
                metadata variables.
                -   key (str):
                        Name of metadata variable.
                -   value (list | tuple):
                        Values of metadata variable across samples. Must have
                        either an index for each sample or a length that is
                        a factor of the length of the dataset. In the latter
                        case, value sequence will be tiled to match the length
                        of the full dataset.
        """
        super(CoreDataset, self).__init__()

        meta = pd.DataFrame(np.arange(n_samples), columns=[self._INDEX])

        # add metadata columns
        for k, v in kwargs.items():
            if isinstance(v, list) or isinstance(v, tuple):
                repeat, remain = divmod(n_samples, len(v))
                if remain != 0 or repeat == 0:
                    raise ValueError(
                        f"{self.__class__.__name__} with {n_samples} samples "
                        + f"got metadata variable {k} with {len(v)} samples, "
                        + f"which are indivisible by {remain} samples.")

                v = list(v) * repeat if repeat > 1 else v

            meta[k] = v

        # create hierarchical index and set instance attributes
        cols = [c for c in meta.columns if c != self._INDEX]
        self.meta = meta if len(cols) == 0 else meta.set_index(
            cols, append=False).sort_index()

    def __len__(
            self
    ):
        """
        Returns:
            (int):
                Number of (metadata, data) pairs stored.
        """
        return self.meta.shape[0]

    def get_metadata(
            self,
            meta_var: str
    ):
        """
        Get values of given metadata variable across all samples.

        Args:
            meta_var (str):
                Name of metadata variable.

        Returns:
            (np.ndarray):
                Values of metadata variable across all samples.
        """
        return self.meta.index.get_level_values(meta_var).to_numpy()

    def filter(
            self,
            pattern: dict[str: list]
    ):
        """
        Get indices of samples with metadata values that match a specified
        pattern.

        Args:
            pattern (dict[str: list]):
                Contains the following key, value pairs:
                -   key (str):
                        Name of metadata variable to filter.
                -   value (list):
                        Values of given metadata variable to include in subset
                        of full dataset.

        Returns:
            (np.ndarray):
                Indices of samples matching the specified pattern.
        """
        mask = None
        for k, v in pattern.items():
            pos = self.meta.index.isin(np.atleast_1d(v), level=k)
            mask = pos if mask is None else (mask & pos)

        return self.meta.loc[mask, self._INDEX].to_numpy()

    def k_fold(
            self,
            n_folds: int,
            meta_vars: list[str] = None,
            sub_idx: np.ndarray[int] = None,
            shuffle: bool = True,
            random_state: int = 42
    ):
        """
        Generate a k-fold split of the underlying dataset, stratified by
        specified metadata variables.

        Args:
            n_folds (int):
                Number of folds to generate.
            meta_vars (list[str], optional):
                Metadata variables against which to stratify splits.
                Defaults to None, in which case data are not stratified.
            sub_idx (list[int] | tuple[int] | np.ndarray[int], optional):
                Subset of dataset indices to consider in split.
                Defaults to None, in which case
            shuffle (bool, optional):
                Whether to shuffle before splitting.
                Defaults to True.
            random_state (int):
                Random seed for reproducibility.
                Defaults to 42.

        Yields:
            i (int):
                Current fold.
            train_idx (np.ndarray):
                Indices for training set.
            test_idx (np.ndarray)
                Indices for test set.
        """
        sub_meta = self.meta.copy()

        # filter values by index
        sub_meta = sub_meta if sub_idx is None else sub_meta.iloc[sub_idx]
        sub_idxs = np.arange(sub_meta.shape[0])

        # create composite metadata variable for stratification
        stratify = (
            np.arange(sub_idxs) if meta_vars is None else
            sub_meta.index.to_frame(index=False)[meta_vars].astype(str).agg(
                "-".join, axis=1).to_numpy())

        # verify adequate data per strata for desired number of groups
        _, counts = np.unique(stratify, return_counts=True)
        if n_folds < 2 or n_folds < np.min(counts):
            raise ValueError(
                f"{self.__class__.__name__}.k_fold expects at least "
                + f"2 valid folds, got n_folds={n_folds} and a distribution "
                + f"of within-strata tallies of {counts}")

        # generate stratified splits
        kf = StratifiedKFold(
            n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        true_idxs = sub_meta[self._INDEX].to_numpy()
        for i, train_idx, test_idx in enumerate(kf.split(sub_idxs, stratify)):
            # convert sub indices back to full dataset indices
            yield i, true_idxs[train_idx], true_idxs[test_idx]

    def get_subset(
            self,
            sub_idx: list[int] | tuple[int] | np.ndarray[int]
    ):
        """
        Get a subset of dataset including only samples with the specified
        indices.

        Args:
            sub_idx (list[int] | tuple[int] | np.ndarray[int]):
                Indices of samples in full dataset to include in subset.

        Returns:
            (Subset):
                Subset of dataset.
        """
        return Subset(self, sub_idx)
