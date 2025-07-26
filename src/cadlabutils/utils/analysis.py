#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July  4 13:06:21 2023
@author: ike
"""

import numpy as np
import sklearn.cluster as skc

import tslearn.clustering as tsc
import tslearn.preprocessing as tsp


"""
"""

MODELS = {
    "kshape": tsc.KShape,
    "timeserieskmeans": tsc.TimeSeriesKMeans,
    "kernelkmeans": tsc.KernelKMeans
}


def summarize(array, start, stop, axis, func=np.mean):
    array = np.take(array, tuple(range(start, stop)), axis=axis)
    array = func(array, axis=axis, keepdims=True)
    return array


def z_score(array, start, stop, axis):
    mean = summarize(array, start, stop, axis)
    std = summarize(array, start, stop, axis, np.std)
    array = (array - mean) / std
    return array


def convolve(data, window):
    data = np.pad(data, window, mode="edge")
    data = np.convolve(data, np.ones(window), mode="same")
    data = data[window:-window]
    data = data / window
    return data


def normalize(data):
    data = data - np.min(data)
    data = data / np.max(data)
    return data


def cluster_time_series(data, model, **kwargs):
    data = tsp.TimeSeriesScalerMeanVariance().fit_transform(data)
    model = MODELS[model](**kwargs)
    model.fit(data)
    return model


def optimize_clustering_inertia(data, model, clusters, **kwargs):
    models = [
        cluster_time_series(data, model, n_clusters=i, **kwargs)
        for i in range(2, clusters + 1)]
    inertias = [model.inertia_ for model in models]
    clusters = inertias.index(min(inertias))
    return models[clusters], clusters + 2
