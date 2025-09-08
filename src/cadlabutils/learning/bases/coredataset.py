#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4 13:06:21 2024
@author: ike
"""


import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


class CoreDataset(Dataset):
    """Store mapping between hierarchical metadata and data indices.

    Class Attributes
    ----------------
    _INDEX : str
        Internal metadata level to track corresponding index of each datum.

    Attributes
    ----------
    meta : pd.DataFrame
        Hierarchically indexed DataFrame storing metadata. Has shape
        (n_samples, n_metadata_variables + 1).
    parent : CoreDataset | None
        If not None, point to another CoreDataset instancefrom which to access
        underlying data.

    Parameters
    ----------
    samples : int
        Number of samples or observations included in dataset.
    parent : CoreDataset, optional
        Another CoreDataset instancefrom which to access underlying data.
        Defaults to None.
    **kwargs
        key : int | str
            Name of metadata variable.
        value (list | tuple):
            Values of metadata variable across samples. Must have a value per
            sample or length that is a factor of the length of the dataset.

    Raises
    ------
    KeyError
        Instantiating from existing metadata DataFrame requires a column of
        integer indices matching `_INDEX` attribute.
    ValueError
        Expanded metadata levels must match number of stored samples.

    Notes
    -----
    `CoreDataset` includes functionality to filter and split a dataset based on
    metadata values. Doing so keep the underlying data intact by instead
    selecting a _subset of data indices to include in a given instance. Child
    class must implement the structure and content of the underlying data
    paired to each index.

    Examples
    --------
    Instantiate with 3 metadata variables.
    >>> # arbitrary metadata values, including constants and lists
    >>> metadata = {
    ... "day": "Mon", "label": ["head", "tail"], "count": [3, 6, 1, 4, 2, 8]}
    >>> test_dataset = CoreDataset(len(metadata["count"]), **metadata)
    >>> test_dataset.meta  # doctest: +NORMALIZE_WHITESPACE
                     _data_index
    day label count
    Mon head  1                2
              2                4
              3                0
        tail  4                3
              6                1
              8                5

    Get all values of metadata variable.
    >>> test_dataset.get_metadata("label")
    array(['head', 'head', 'head', 'tail', 'tail', 'tail'], dtype=object)

    Get data with set metadata values.
    >>> test_idx = test_dataset.filter({"label": ["head"]})
    >>> test_idx.meta  # doctest: +NORMALIZE_WHITESPACE
                     _data_index
    day label count
    Mon head  1                2
              2                4
              3                0

    Generate k-fold split stratified by label metadata.
    >>> for i, (t, v) in enumerate(
    ...         test_dataset.k_fold(3, meta_vars=["label"])):
    ...     print(f"fold {i}:")
    ...     t.meta  # doctest: +NORMALIZE_WHITESPACE
    ...     v.meta  # doctest: +NORMALIZE_WHITESPACE
    ...     print("-" * 28)
    fold 0:
                     _data_index
    day label count
    Mon head  2                4
              3                0
        tail  4                3
              6                1
                     _data_index
    day label count
    Mon head  1                2
        tail  8                5
    ----------------------------
    fold 1:
                     _data_index
    day label count
    Mon head  1                2
              3                0
        tail  6                1
              8                5
                     _data_index
    day label count
    Mon head  2                4
        tail  4                3
    ----------------------------
    fold 2:
                     _data_index
    day label count
    Mon head  1                2
              2                4
        tail  4                3
              8                5
                     _data_index
    day label count
    Mon head  3                0
        tail  6                1
    ----------------------------
    """
    _INDEX = "_data_index"

    def __init__(
            self,
            samples: int | pd.DataFrame,
            parent: Dataset = None,
            **kwargs
    ):
        super(CoreDataset, self).__init__()
        self.parent = parent


        # transfer existing metadata
        if isinstance(samples, pd.DataFrame):
            if self._INDEX not in samples.columns:
                raise KeyError(
                    f"DataFrame `samples` must have a {self._INDEX} column, "
                    + f"but none found.")

            self.meta = samples
            return

        # add metadata columns
        meta = pd.DataFrame(np.arange(samples), columns=[self._INDEX])
        for k, v in kwargs.items():
            if any([isinstance(v, t) for t in (list, tuple)]):
                repeat, remain = divmod(samples, len(v))
                if remain != 0 or repeat == 0:
                    raise ValueError(
                        f"{self.__class__.__name__} with {samples} samples "
                        + f"got metadata variable {k} with {len(v)} samples, "
                        + f"which are indivisible by {remain} samples.")

                v = list(v) * repeat if repeat > 1 else v

            meta[k] = v
            meta[k] = meta[k].astype("category")

        # create hierarchical index and set instance attributes
        cols = [c for c in meta.columns if c != self._INDEX]
        self.meta = meta if len(cols) == 0 else meta.set_index(
            cols, append=False).sort_index()

    def __len__(
            self
    ):
        """Get length of instance.

        Returns
        -------
        int
            Number of data points stored.
        """
        return self.meta.shape[0]

    def __getitem__(
            self,
            idx
    ):
        """Get index of underlying data.

        Parameters
        ----------
        idx : int
            0-indexed index of item.

        Returns
        -------
        item : int | Any
            Item stored at `idx`. If `parent` attribute is None, `item` is an
            ``int`` that corresponds to the index of the `idx`th sample stored
            in `meta` attribute. If `parent` attribute is not None, `item` is
            the result of `parent.__getitem__(index of the `idx`th sample)`.

        Notes
        -----
        There are two sets of indices to speak of -- indices that point to each
        sample stored in current instance, and indices that point to the
        underlying data to which each set of metadata values refer. The former
        is 0-indexed up to the length of the instance. The latter values are
        stored in `meta[CoreDataset._INDEX]`.
        `CoreDataset.__getitem__` takes the former index system as input.
        """
        idx = self.meta.iloc[idx][self._INDEX]
        item = idx if self.parent is None else self.parent[idx]
        return item

    def _subset(
            self,
            sub_idx: list[int] | tuple[int] | np.ndarray[int]
    ):
        """Get a _subset of dataset with the specified indices.

        Parameters
        ----------
        sub_idx : list[int] | tuple[int] | np.ndarray[int]
            Indices of samples in full dataset to include in _subset. Indices
            correspond to those in `meta[CoreDataset._INDEX]`.

        Returns
        -------
        _subset : CoreDataset
            Subset of dataset.
        """
        subset = CoreDataset(
            self.meta[self.meta[self._INDEX].isin(sub_idx)],
            parent=self if self.parent is None else self.parent)
        return subset


    def get_metadata(
            self,
            meta_var: str
    ):
        """Get values of given metadata variable across all samples.

        Parameters
        ----------
        meta_var : str
            Name of metadata variable.

        Returns
        -------
        np.ndarray
            Values of metadata variable across all samples.
        """
        return self.meta.index.get_level_values(meta_var).to_numpy()

    def filter(
            self,
            pattern: dict[str: list]
    ):
        """Get indices of samples with specific metadata values.

        Parameters
        ----------
        pattern : dict[str: list]
            key : str
                Name of metadata variable to filter.
            value : list
                Values of given metadata variable to include in _subset of full
                dataset.

        Returns
        -------
        CoreDataset
            New instance with samples matching the specified pattern.
        """
        mask = None
        for k, v in pattern.items():
            pos = self.meta.index.isin(np.atleast_1d(v), level=k)
            mask = pos if mask is None else (mask & pos)

        sub_idx = self.meta.loc[mask, self._INDEX].to_numpy()
        return self._subset(sub_idx)

    def k_fold(
            self,
            n_folds: int,
            meta_vars: list[str] = None,
            shuffle: bool = True,
            random_state: int = 42
    ):
        """Stratify data index splits based on underlying metadata.

        Parameters
        ----------
        n_folds : int
            Number of folds to generate.
        meta_vars : list[str], optional
            Metadata variables against which to stratify splits.
            Defaults to None, in which case data are not stratified.
        shuffle : bool, optional
            Whether to shuffle before splitting.
            Defaults to True.
        random_state : int
            Random seed for reproducibility.
            Defaults to 42.

        Yields
        ------
        t_set : CoreDataset
            Data for current training split.
        v_set : CoreDataset
            Data for current test set.

        Raises
        ------
        ValueError
            Insufficient data diversity to split unique groups in `meta_vars`
            into `n_folds` folds.
        """
        sub_meta = self.meta.copy()

        # create composite metadata variable for stratification
        stratify = (
            np.ones(len(self)) if meta_vars is None else
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
        for tdx, vdx in kf.split(np.zeros(len(self)), stratify):
            t_set = self._subset(self.meta.iloc[tdx][self._INDEX].to_numpy())
            v_set = self._subset(self.meta.iloc[vdx][self._INDEX].to_numpy())
            yield t_set, v_set
