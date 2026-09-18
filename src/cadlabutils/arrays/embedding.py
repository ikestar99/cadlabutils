#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July 4 13:06:21 2023
@author: ike
"""


# 1. Standard library imports
import random

# 2. Third-party library imports
import numpy as np
import scipy.ndimage as scn
import scipy.stats as sst
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


rng = np.random.default_rng(42)


def shuffle_pca_variance(
        arr: np.ndarray,
        repeat: int = 10,
        shuffle_rows: bool = True,
        shuffle_cols: bool = True,
):
    """Generate a shuffled variance distribution with PCA.

    Parameters
    ----------
    arr : np.ndarray
        Data to analyze with PCA. Has shape (n_observations, n_features).
    repeat : int, optional
        Number of null replicates to generate.
        Defaults to 10.
    shuffle_rows : bool, optional
        If True, shuffle features within observations. If False, `shuffle_cols`
        must be True.
        Defaults to True.
    shuffle_cols : bool, optional
        If True, shuffle observations within features. If False, `shuffle_rows`
        must be True.
        Defaults to True.

    Returns
    -------
    np.ndarray
        Ratio of total variance in 'arr' explained by each principal component
        following PCA after shuffling. Has shape ('repeat', n_components).
    """
    if not shuffle_rows and not shuffle_cols:
        raise ValueError(
            "`shuffle_rows` and `shuffle_cols` cannot both be False.")

    ratios = []
    for _ in range(repeat):
        shuffled = arr.copy()
        shuffled = shuffled if not shuffle_rows else np.stack(
            [rng.permutation(row) for row in arr], axis=0)
        shuffled = shuffled if not shuffle_cols else np.stack(
            [rng.permutation(row) for row in arr.T], axis=1)
        ratios.append(PCA().fit(shuffled).explained_variance_ratio_)

    return np.stack(ratios, axis=0)


def shuffle_cca_correlation(
        arr_1: np.ndarray,
        arr_2: np.ndarray,
        n_correlates: int,
        repeat: int = 10
):
    """Generate a shuffled correlation distribution with CCA.

    Parameters
    ----------
    arr_1 : np.ndarray
        First dataset to correlate with CCA. Has shape
        (n_observations, n_features_1).
    arr_2 : np.ndarray
        Second dataset to correlate with CCA. Has shape
        (n_observations, n_features_2).
    n_correlates : int
        Number of canonical correlates to generate.
    repeat : int, optional
        Number of null replicates to generate.
        Defaults to 10.

    Returns
    -------
    np.ndarray
        Correlations between canonical correlates following CCA after shuffling
        correspondence between observations in `arr_1` and `ar_2`. Has shape
        ('repeat', n_correlates).
    """
    if arr_1.shape[0] != arr_2.shape[0]:
        raise ValueError(
            "`arr_1` and `arr_2` must have the same number of observations.")

    correlations = []
    for _ in range(repeat):
        _arr_2 = arr_2.copy()[rng.permutation(np.arange(arr_1.shape[0]))]
        _cca = CCA(n_components=n_correlates)
        _cca.fit(arr_1.copy(), _arr_2.copy())
        _arr_1, _arr_2 = _cca.transform(arr_1.copy(), _arr_2)
        correlations.append([
            np.corrcoef(_arr_1[:, i], _arr_2[:, i])[0, 1]
            for i in range(_arr_1.shape[1])])

    return np.stack(correlations, axis=0)


def pca(
        arr: np.ndarray,
        repeat: int = 10,
        **kwargs
):
    raw_pca = PCA(n_components=None).fit(arr.copy())
    signed_contrib = np.ascontiguousarray(
        raw_pca.components_ * np.abs(raw_pca.components_)
        * raw_pca.explained_variance_ratio_[:, None])
    null_var = None if repeat <= 1 else shuffle_pca_variance(
        arr, repeat=repeat, **kwargs)
    return raw_pca, raw_pca.explained_variance_ratio_, signed_contrib, null_var


def cca(
        arr_1: np.ndarray,
        arr_2: np.ndarray,
        n_correlates: int,
        repeat: int = 10,
):
    raw_cca = CCA(n_components=None)
    raw_cca.fit(arr_1.copy(), arr_2.copy())
    _arr_1, _arr_2 = raw_cca.transform(arr_1.copy(), arr_2.copy())
    corr = np.array([
        np.corrcoef(_arr_1[:, i], _arr_2[:, i])[0, 1]
        for i in range(_arr_1.shape[1])])
    null_corr = None if repeat <= 1 else shuffle_cca_correlation(
        arr_1, arr_2, n_correlates=n_correlates, repeat=repeat)
    return raw_cca, corr, null_corr
