#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4 13:06:21 2024
@author: ike
"""


import numpy as np
import pandas as pd
import operator

from torch.utils.data import Dataset, Subset, ConcatDataset

from sklearn.model_selection import StratifiedKFold


OPERATIONS = {
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne
}


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

    Examples:
    ---------------------------------------------------------------------------
        Instantiate with 1d datum.
        >>> # test_data shape = (3 data points, 5 observations per datum)
        >>> test_data = np.arange(15).reshape(3, 5)
        >>> # test_meta shape = (3 data points, 2 metadata variables)
        >>> test_meta = pd.DataFrame(
        ... {"day": "Mon", "label": [1, 1, 2]})
        >>> # test_axes = number of axes in each datum, 1 if datum is 1d array
        >>> test_axes = 1
        >>> test_dataset = CoreDataset(test_data, test_meta, test_axes)
        >>> test_dataset.data
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14]])
        >>> test_dataset.meta
                                   _data_index
        day label _repeated_trial
        Mon 1     0                          0
                  1                          1
            2     2                          2

        -----------------------------------------------------------------------
        Instantiate using "expand" arg.
        Note that an internal _repeated trial" metadata level distinguishes
        between repetitions of the same set of metadata values.
        >>> # test_data shape = (3 * 2 data points, 5 observations per datum)
        >>> test_data = np.arange(30).reshape(3, 2, 5)
        >>> test_expand = [(1, "channel", ["blue", "red"])]
        >>> test_dataset = CoreDataset(
        ... test_data, test_meta, test_axes, test_expand)
        >>> test_dataset.data
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24],
               [25, 26, 27, 28, 29]])
        >>> test_dataset.meta
                                           _data_index
        day label _repeated_trial channel
        Mon 1     0               blue               0
                                  red                1
                  1               blue               2
                                  red                3
            2     2               blue               4
                                  red                5

        -----------------------------------------------------------------------
        Filter instance using __getitem__.
        >>> test_slice = test_dataset[1:6:2]
        >>> test_slice.data
        array([[ 5,  6,  7,  8,  9],
               [15, 16, 17, 18, 19],
               [25, 26, 27, 28, 29]])
        >>> test_slice.meta
                                           _data_index
        day label _repeated_trial channel
        Mon 1     0               red                0
                  1               red                1
            2     2               red                2
        >>> test_slice = test_dataset[{"label": {">": 0}, "channel": ["blue"]}]
        >>> test_slice.data
        array([[ 0,  1,  2,  3,  4],
               [10, 11, 12, 13, 14],
               [20, 21, 22, 23, 24]])
        >>> test_slice.meta
                                           _data_index
        day label _repeated_trial channel
        Mon 1     0               blue               0
                  1               blue               1
            2     2               blue               2

        -----------------------------------------------------------------------
        Add two instances together.
        Note that "label" and "channel" metadata levels are dropped after
        addition since they aren't shared across operands.
        >>> test_data = np.arange(10).reshape(2, 5)
        >>> test_meta_1 = pd.DataFrame({"day": "Mon", "label": [1, 2]})
        >>> test_meta_2 = pd.DataFrame({"day": "Tue", "channel": [5, 6]})
        >>> test_axes = 1
        >>> test_dataset_1 = CoreDataset(test_data, test_meta_1, test_axes)
        >>> test_dataset_2 = CoreDataset(-test_data, test_meta_2, test_axes)
        >>> test_add = test_dataset_1 + test_dataset_2
        >>> test_add.data
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [ 0, -1, -2, -3, -4],
               [-5, -6, -7, -8, -9]])
        >>> test_add.meta
                             _data_index
        _repeated_trial day
        0               Mon            0
        1               Mon            1
        2               Tue            2
        3               Tue            3

        -----------------------------------------------------------------------
        Extract scalar summary statistic from each datum.
        >>> test_dataset.data
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24],
               [25, 26, 27, 28, 29]])
        >>> test_dataset.extract_statistic(np.mean, "mean")
                                           _data_index  mean
        day label _repeated_trial channel
        Mon 1     0               blue               0   2.0
                                  red                1   7.0
                  1               blue               2  12.0
                                  red                3  17.0
            2     2               blue               4  22.0
                                  red                5  27.0

        -----------------------------------------------------------------------
        Apply function across channels
        >>> test_agg = test_dataset.apply_function(
        ... np.mean, ["label"], axis=0)
        >>> test_agg.data
        array([[10., 11., 12., 13., 14.],
               [15., 16., 17., 18., 19.]])
        >>> test_agg.meta
                                     _data index
        day channel _repeated trial
        Mon blue    0                          0
            red     1                          1
        >>> test_agg = test_dataset.apply_function(
        ... np.median, ["label", "channel"], axis=0)
        >>> test_agg.data
        array([[ 2.5,  3.5,  4.5,  5.5,  6.5],
               [12.5, 13.5, 14.5, 15.5, 16.5],
               [22.5, 23.5, 24.5, 25.5, 26.5]])
        >>> test_agg.meta
                             _data index
        day _repeated trial
        Mon 0                          0
            1                          1
            2                          2
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

        meta = pd.DataFrame(np.arange(n_samples)[..., None])

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
            cols, append=False)

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

    def filter_idx(
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
        sub_meta = self.meta.copy()
        for k, v in pattern.items():
            sub_meta = sub_meta[sub_meta.index.isin(np.atleast_1d(v), level=k)]

        return sub_meta[self._INDEX].to_numpy()

    def kfold_split_idx(
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
                f"{self.__class__.__name__}.kfold_split_idx expects at least "
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
