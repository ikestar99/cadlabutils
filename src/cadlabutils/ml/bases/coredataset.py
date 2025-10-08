#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4 13:06:21 2024
@author: ike
"""


# 1. Standard library imports
from pathlib import Path

# 2. Third-party library imports
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

# 3. Local application / relative imports
import cadlabutils as cdu


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
    truth_var : str
        Metadata variable to use as ground truth label for classification.
        Defaults to None.
    parent : CoreDataset | None
        Parent CoreDataset instance for which the current instance is a
        filtered view. Mediates access to underlying data.
        Defaults to None.

    Parameters
    ----------
    samples : int | str | Path | pd.DataFrame
        If ``int``, number of samples or observations included in dataset.
        If ``str`` or ``Path``, points to a file containing saved metadata.
        If ``pd.DataFrame``, a filtered subset of `meta` attribute from parent
        instance.
    truth_var : str, optional
        Metadata variable to use as ground truth label for classification.
        Defaults to None.
    _parent : CoreDataset, optional
        Another CoreDataset instancefrom which to access underlying data.
        Defaults to None.
    **kwargs
        key : int | str
            Name of metadata variable.
        value (str | float | list | tuple | np.ndarray):
            Values of metadata variable across samples. Iterable must have a
            index per sample or a length that can be tiled to fit size of
            dataset. Scalar will assign the same value to all samples.

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
    ... "day": "Mon", "label": ["head", "tail"], "count": [3, 6, 1, 4, 2, 8],
    ... "truth_var": "label"}
    >>> t_dataset = CoreDataset(len(metadata["count"]), **metadata)
    >>> t_dataset.meta  # doctest: +NORMALIZE_WHITESPACE
                     _data_index
    day label count
    Mon head  1                2
              2                4
              3                0
        tail  4                3
              6                1
              8                5

    Rich print instance data.
    >>> print(t_dataset)  # doctest: +NORMALIZE_WHITESPACE
    ┏━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
    ┃ day ┃ count ┃ class: head ┃ class: tail ┃
    ┡━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
    │ Mon │ 1     │ 1           │             │
    │     │ 2     │ 1           │             │
    │     │ 3     │ 1           │             │
    │     │ 4     │             │ 1           │
    │     │ 6     │             │ 1           │
    │     │ 8     │             │ 1           │
    │     │       │ 3           │ 3           │
    └─────┴───────┴─────────────┴─────────────┘

    Get all values of metadata variable.
    >>> t_dataset.get_metadata("label")
    array(['head', 'head', 'head', 'tail', 'tail', 'tail'], dtype=object)

    Get all values of multiple metadata variables.
    >>> t_dataset.get_metadata(["label", "count"])
    array(['head-1', 'head-2', 'head-3', 'tail-4', 'tail-6', 'tail-8'],
          dtype=object)

    Filter data with set metadata values.
    >>> t_idx = t_dataset.filter(label=["head"])
    >>> t_idx.meta  # doctest: +NORMALIZE_WHITESPACE
                     _data_index
    day label count
    Mon head  1                2
              2                4
              3                0

    Balance metadata values to set ratio.
    >>> t_subset = t_dataset.balance_metadata(
    ...     meta_var=["label"], balance={"head": 1.5, "tail": 0.5})
    >>> t_subset.meta  # doctest: +NORMALIZE_WHITESPACE
                     _data_index
    day label count
    Mon head  1                2
              2                4
              3                0
        tail  6                1

    Generate k-fold split stratified by label metadata.
    >>> for i, (t, v) in enumerate(
    ...         t_dataset.k_fold(3, stratify=["label"])):
    ...     print(f"fold {i} train:")
    ...     t.meta  # doctest: +NORMALIZE_WHITESPACE
    ...     print(f"fold {i} valid:")
    ...     v.meta  # doctest: +NORMALIZE_WHITESPACE
    ...     print("- " * 14)
    fold 0 train:
                     _data_index
    day label count
    Mon head  2                4
              3                0
        tail  4                3
              8                5
    fold 0 valid:
                     _data_index
    day label count
    Mon head  1                2
        tail  6                1
    - - - - - - - - - - - - - -
    fold 1 train:
                     _data_index
    day label count
    Mon head  1                2
              3                0
        tail  4                3
              6                1
    fold 1 valid:
                     _data_index
    day label count
    Mon head  2                4
        tail  8                5
    - - - - - - - - - - - - - -
    fold 2 train:
                     _data_index
    day label count
    Mon head  1                2
              2                4
        tail  6                1
              8                5
    fold 2 valid:
                     _data_index
    day label count
    Mon head  3                0
        tail  4                3
    - - - - - - - - - - - - - -
    """
    _INDEX = "_data_index"

    def __init__(
            self,
            samples: int | str | Path | pd.DataFrame,
            truth_var: str = None,
            _parent: Dataset = None,
            **kwargs
    ):
        super(CoreDataset, self).__init__()
        self.parent = _parent
        self.truth_var = truth_var

        if type(samples) in (Path, str):  # load existing metadata
            samples = pd.read_parquet(Path(samples))
        if isinstance(samples, pd.DataFrame):  # transfer existing metadata
            if self._INDEX not in samples.columns:
                raise KeyError(
                    f"DataFrame `samples` must have a {self._INDEX} column, "
                    + f"but none found.")

            self.meta = samples.sort_index()
        else:
            self.meta = self._make_frame(samples, **kwargs)

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
            Index of sample to pull.

        Returns
        -------
        item : pd.Series | Any
            sample stored at `idx`. If `parent` attribute is None, `item` is a
            ``Series`` that contains all metadata for the `idx`th sample stored
            in the instance. If `parent` attribute is not None, `item` is the
            result of `parent.__getitem__(index of the `idx`th sample)`.

        Notes
        -----
        There are two sets of indices to speak of -- indices that point to each
        sample stored in current instance, and indices that point to the
        underlying data to which each set of metadata values refer. The former
        is 0-indexed up to the length of the instance. The latter values are
        stored in `meta[CoreDataset._INDEX]`.
        `CoreDataset.__getitem__` takes the former index system as input.
        """
        item = self.meta.iloc[[idx]].reset_index(drop=False).iloc[0]
        item = item if self.parent is None else self.parent[item[self._INDEX]]
        return item

    def __add__(
            self,
            other
    ):
        """Add two instances together.

        Parameters
        ----------
        other : CoreDataset
            Instance to add to current instance.

        Returns
        -------
        added : CoreDataset
            New instance with combined deduplicated metadata.
        """
        if type(other) is not CoreDataset:
            raise ValueError(
                f"Cannot add {other.__class__.__name__} to CoreDataset.")
        elif other.parent != self.parent or self.parent is None:
            raise ValueError("Summed CoreDatasets must have the same parent.")
        elif self.meta.index.names != other.meta.index.names:
            raise KeyError(
                "Summed CoreDatasets must have the same metadata variables")

        add_meta = pd.concat([self.meta, other.meta]).drop_duplicates(
            subset=self._INDEX, keep="first")
        added = CoreDataset(
            add_meta, truth_var=self.truth_var, _parent=self.parent)
        return added

    def __radd__(
            self,
            other
    ):
        """Add two instances together. See __add__ for more information."""
        return self.__add__(other)

    def __iadd__(
            self,
            other
    ):
        """Add two instances in-place. See __add__ for more information."""
        self.meta = (self + other).meta
        return self

    def __str__(
            self
    ):
        """Get string representation of instance.

        Returns
        -------
        str
            Summary table of stored samples per metadata combination.
        """
        table = self.meta
        if self.truth_var is not None:
            table = table.pivot_table(
                index=[
                    c for c in table.index.names if c != self.truth_var],
                columns=self.truth_var, values=self._INDEX, aggfunc="count",
                fill_value=0, observed=True).reset_index()
            table.columns = [
                c if c in self.meta.index.names else f"class: {c}"
                for c in table.columns]
            count_cols = [
                c for c in table.columns if c not in self.meta.index.names]
            rem_cols = [
                c for c in table.columns if c in self.meta.index.names]
            totals = table[count_cols].sum().to_frame().T
            totals[rem_cols] = ""  # leading columns empty
            table = table.astype(str)
            table[rem_cols] = table[rem_cols].where(
                ~table[rem_cols].eq(table[rem_cols].shift()), "")
            table[count_cols] = table[count_cols].replace(
                to_replace="0", value="", regex=False)
            table = pd.concat(
                [table, totals[table.columns]], ignore_index=True)

        console = cdu.Console()
        with console.capture() as capture:
            console.print(cdu.get_rich_table(table))

        return capture.get()

    @classmethod
    def _make_frame(
            cls,
            n: int,
            **kwargs
    ):
        # add metadata columns
        meta = pd.DataFrame(np.arange(n), columns=[cls._INDEX])
        for k, v in kwargs.items():
            if any([isinstance(v, t) for t in (list, tuple, np.ndarray)]):
                repeat, remain = divmod(n, len(v))
                if remain != 0 or repeat == 0:
                    raise ValueError(
                        f"{cls.__name__} with {n} samples got metadata "
                        + f"variable {k} with {len(v)} samples, which are "
                        + f"indivisible by {remain} samples.")

                v = list(v) * repeat if repeat > 1 else v

            meta[k] = v
            meta[k] = meta[k].astype("category")

        # create hierarchical index and set instance attributes
        cols = [c for c in meta.columns if c != cls._INDEX]
        meta = meta if len(cols) == 0 else meta.set_index(
            cols, append=False).sort_index()
        return meta

    def _save(
            self,
            meta_csv: Path
    ):
        """Save metadata in compressed parquet format.

        Parameters
        ----------
        meta_csv : Path
            Path in which to save metadata (.parquet).
        """
        meta_csv = meta_csv.with_suffix(".parquet")
        self.meta.to_parquet(meta_csv, index=True, compression="zstd")

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
        subset : CoreDataset
            Subset of dataset.
        """
        subset = CoreDataset(
            self.meta[self.meta[self._INDEX].isin(sub_idx)].copy(),
            truth_var=self.truth_var,
            _parent=self if self.parent is None else self.parent)
        return subset

    def get_metadata(
            self,
            meta_var: str | list[str]
    ):
        """Get values of given metadata variable(s) across all samples.

        Parameters
        ----------
        meta_var : str | list[str]
            Name of metadata variable(s).

        Returns
        -------
        np.ndarray
            Values of metadata variable across all samples. If `meta_var` is a
            list, metadata values are spliced together as type ``str`` for each
            sample with "-" as a separator.

        Notes
        -----
        If `meta_var` has multiple entries, they are sorted in the order that
        they appear in instance hierarchical index.
        """
        if type(meta_var) is str or len(meta_var) == 1:
            meta_var = meta_var[0] if isinstance(meta_var, list) else meta_var
            metadata = self.meta.index.get_level_values(meta_var).to_numpy()
        else:
            meta_var = sorted(
                meta_var, key=lambda n: self.meta.index.names.index(n))
            metadata = self.meta.index.to_frame(index=False)[meta_var].astype(
                str).agg("-".join, axis=1).to_numpy()

        return metadata

    def add_metadata(
            self,
            **kwargs
    ):
        """Add metadata level(s) to dataset.

        Parameters
        ----------
        **kwargs
            key : int | str
                Name of metadata variable to add. Must not already exist.
            value (str | float | list | tuple | np.ndarray):
                Values of metadata variable across samples. Iterable must have a
                index per sample. Scalar will assign the same value to all
                samples.

        Raises
        ------
        KeyError
            Metadata variable name already exists in instance data.

        Notes
        -----
        To use new metadata variable as ground truth, update `truth_var`
        attribute.
        """
        for k, v in kwargs.items():
            if k in self.meta.index.names:
                raise KeyError(f"{k} metadata already exists in dataset.")

            self.meta[k] = v

        self.meta = self.meta.set_index([k for k in kwargs], append=True)

    def balance_metadata(
            self,
            meta_var: str | list[str],
            balance: dict[int | float | str | bool, float] = None
    ):
        """Normalize unique metadata prevalence to fixed ratio.

        Parameters
        ----------
        meta_var : str | list[str]
            Name of metadata variable(s) to balance.
        balance : dict[int | float | str | bool, float], optional
            key : int | float | str | bool
                Unique combination of `meta_var` metadata values. If `meta_var`
                is ``str`` or ``list`` with a single entry, `balance` keys must
                be a value found in this metadata colum. Otherwise, keys must
                be values found after splicing ``str`` metadata together.
            value : float
                Relative proportion of specified metadata value to keep in
                returned dataset. Unique metadata values not listed in
                `balance` are omitted from output dataset.
            Defaults to None, in which case all unique combinations of metadata
            values are set to equal prevalence.

        Returns
        -------
        CoreDataset
            Balanced subset of dataset.

        See Also
        --------
        get_metadata : Used to splice metadata values within sample.
        """
        # calculate existing distribution
        metadata = self.get_metadata(meta_var)
        counts = {k: v for k, v in
                  zip(*np.unique(metadata, return_counts=True))}

        # filter out unneeded labels
        balance = {k: 1 for k in counts} if balance is None else balance
        counts = {k: v for k, v in counts.items() if k in balance}

        # compute limiting factor --> target counts
        limiting = min(counts[k] / balance[k] for k in balance)
        new_counts = {k: int(limiting * balance[k]) for k in balance}

        # change class proportions to match calculated distribution
        mask = np.zeros(len(metadata), dtype=bool)

        # randomly sample desired proportion of each unique value
        rng = np.random.default_rng(seed=42)
        for k, c in new_counts.items():
            indices = rng.choice(
                np.nonzero(metadata == k)[0], size=c, replace=False)
            mask[indices] = True

        return self._subset(self.meta.loc[mask, self._INDEX].to_numpy())

    def filter(
            self,
            **kwargs
    ):
        """Get indices of samples with specific metadata values.

        Parameters
        ----------
        **kwargs
            key : str
                Name of metadata variable to filter.
            value : list | tuple | np.ndarray
                Values of given metadata variable to include in _subset of full
                dataset.

        Returns
        -------
        CoreDataset
            New instance with samples matching the specified pattern.

        Notes
        -----
        Using the key "truth_var" will automatically filter the metadata
        variable used as ground truth labels.
        """
        mask = None
        for k, v in kwargs.items():
            k = self.truth_var if k == "truth_var" else k
            pos = self.meta.index.isin(v, level=k)
            mask = pos if mask is None else (mask & pos)

        sub_idx = self.meta.loc[mask, self._INDEX].to_numpy()
        return self._subset(sub_idx)

    def k_fold(
            self,
            n_folds: int,
            stratify: list[str] = None,
            grouping: list[str] = None,
            shuffle: bool = True,
            random_state: int = 42
    ):
        """Stratify data index splits based on underlying metadata.

        Parameters
        ----------
        n_folds : int
            Number of folds to generate.
        stratify : list[str], optional
            Metadata variables against which to stratify splits.
            Defaults to None, in which case data are not stratified.
        grouping : list[str], optional
            Metadata variables against which to group splits.
            Defaults to None, in which case data are not grouped.
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

        Notes
        -----
        `stratify` and `grouping` metadata variables serve different purposes.
        -   With `stratify`, each unique combination of variables appears in
            both training and validation subsets in roughly equal proportion,
            yielding a split wherein data heterogeneity is preserved.
        -   With `grouping`, each unique combination of variables (a "group")
            appears only in one subset, ex. in the training but not validation
            split. As a consequence, each group serves in exactly one
            validation set across all folds.

        Thus, `stratify` is used to ensure that the validation split resembles
        the training split, while `grouping` is used to evaluate if a given
        model generalizes to data excluded from training.
        """
        # create composite metadata variable for stratification and grouping
        strat = np.ones(len(self)) if stratify is None else self.get_metadata(
            stratify)
        group = (
            np.arange(len(self)) if grouping is None else self.get_metadata(
                grouping))

        # verify adequate data per strata for desired number of groups
        _, counts = np.unique(stratify, return_counts=True)
        if n_folds < 2 or n_folds < np.min(counts):
            raise ValueError(
                f"{self.__class__.__name__}.k_fold expects at least "
                + f"2 valid folds, got n_folds={n_folds} and a distribution "
                + f"of within-strata tallies of {counts}")

        # generate stratified splits
        kf = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        for tdx, vdx in kf.split(np.zeros(len(self)), strat, group):
            t_set = self._subset(self.meta.iloc[tdx][self._INDEX].to_numpy())
            v_set = self._subset(self.meta.iloc[vdx][self._INDEX].to_numpy())
            yield t_set, v_set
