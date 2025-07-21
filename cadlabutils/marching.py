#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 09:00:00 2025
@origin: https://github.com/rhngla/topo-preserve-fastmarching
"""


import numpy as np
import scipy.ndimage as scn

import scipy.io as sio
import matplotlib.pyplot as plt


"""
Fast marching algorithm for volumetric label generation
===============================================================================
"""


def _topo(
        voxel: np.ndarray,
        faces_only: bool
):
    """
    Computes topological numbers for the central point of an image patch.
    """
    voxel = voxel.copy()
    if faces_only:
        # Number of 6-adjacent components in the 18-neighborhood of the center
        faces = scn.generate_binary_structure(voxel.ndim, 1)
        edges = scn.generate_binary_structure(voxel.ndim, voxel.ndim - 1)
        components, _ = scn.label(voxel * edges, structure=np.ones((3, 3, 3)))
        components = components * faces
    else:
        # Number of 26-components in the 26-neighborhood of the center
        voxel[tuple([1 for _ in range(voxel.ndim)])] = 0
        components, _ = scn.label(voxel, structure=np.ones((3, 3, 3)))

    return np.unique(components[components > 0]).size == 1


def fast_march(
        stack: np.ndarray,  # <x, y, z>
        seeds: np.ndarray,  # <points, coordinates>
        d_max: float,
        aniso: list[float],  # [x, y, z] anisotropy
        pad: int = 2,
        t_max: float = np.inf
):
    """
    Fast Marching Topology Check
    """
    aniso = [max(i, 0) for i in aniso]
    aniso = np.ones(3) if sum(aniso) == 0 else np.array(aniso)
    aniso = np.repeat(aniso / np.linalg.norm(aniso), 2) ** 2

    stack = np.pad(stack, pad, mode="constant")
    seeds = np.ravel_multi_index(
        np.round(seeds).astype(int).T + pad, stack.shape)
    f_idx = np.array([-1, 1] * 3) * np.repeat(
        [np.prod(stack.shape[1:]), stack.shape[-1], 1],2)

    t = np.full(stack.shape, np.inf, dtype=np.float16)
    d = np.full(stack.shape, np.inf, dtype=np.float16)
    kt = np.zeros(stack.shape, dtype=np.uint8)
    t.flat[seeds], d.flat[seeds], kt.flat[seeds] = 0, 0, 1
    ktcopy = kt.copy()

    complex_points = []
    nhood = np.concatenate([seeds + idx for idx in f_idx])
    trial_points, t_trial_points = np.array([], dtype=int), np.array([])
    while True:
        nhood = np.unique(nhood[(nhood < stack.size)])
        nhood = nhood[(kt.flat[nhood] != 1) & (stack.flat[nhood] != 0)]
        if nhood.size > 0:
            ind = np.stack([nhood] * f_idx.size, axis=-1) + f_idx[None]  # 2d
            kt_1 = kt.flat[ind] != 1

            tpms, dpms = t.flat[ind], d.flat[ind]  # both 2d
            tpms[kt_1], dpms[kt_1] = np.inf, np.inf
            tpms.sort(axis=1)
            dpms.sort(axis=1)

            h = aniso[np.argsort(tpms, axis=1)]  # 2d
            h_cum = np.cumsum(h, axis=1)  # 2d
            ct = 1 / stack.flat[nhood] ** 2
            tpms_cum = np.cumsum(tpms * h, axis=1)  # 2d
            tpms_2_cum = np.cumsum(tpms ** 2 * h, axis=1)  # 2d
            nt = np.sum(
                (tpms_2_cum - tpms_cum ** 2 / h_cum) <= ct[:, None],
                axis=1)  # 1d
            if np.sum(aniso == 0) > 0:
                nt[nt == 0] = 1

            tdx = np.arange(tpms.shape[0]) + ((nt - 1) * tpms.shape[0])  # 1d
            nt_h = h_cum.T.flat[tdx]  # 1d

            step = tpms_2_cum.T.flat[tdx] * nt_h - tpms_cum.T.flat[tdx] ** 2
            t.flat[nhood] = (
                (tpms_cum.T.flat[tdx] + np.sqrt(ct * nt_h - step)) / nt_h)

            n = np.ones(tpms.shape) * np.arange(tpms.shape[1])[None] + 1
            dpms_cum = np.cumsum(dpms, axis=1)
            dpms_2_cum = np.cumsum(dpms ** 2, axis=1)
            nd = np.sum((dpms_2_cum - dpms_cum ** 2 / n) <= 1, axis=1)
            ddx = np.arange(dpms.shape[0]) + ((nd - 1) * dpms.shape[0])

            step = dpms_2_cum.T.flat[ddx] * nd - dpms_cum.T.flat[ddx] ** 2
            d.flat[nhood] = (dpms_cum.T.flat[ddx] + np.sqrt(nd - step)) / nd

            keep = nhood[(kt.flat[nhood] == 0) & ~np.isinf(t.flat[nhood])]
            kt.flat[keep] = 2
            trial_points = np.concatenate([trial_points, keep])
            t_trial_points = np.concatenate([t_trial_points, t.flat[keep]])

        if trial_points.size == 0:
            break

        min_ind = np.argmin(t_trial_points)
        new_point = trial_points[min_ind]
        nhood = new_point + f_idx
        if np.isinf(t_trial_points[min_ind]):
            break

        kt.flat[new_point], ktcopy.flat[new_point] = 1, 1
        t_trial_points[min_ind] = np.inf

        x, y, z = np.unravel_index(new_point, stack.shape)
        check = ktcopy[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2] == 1
        if not (_topo(check, True) and _topo(1 - check, False)):
            complex_points += [new_point]
            ktcopy.flat[new_point] = 0
        if t.flat[new_point] >= t_max or d.flat[new_point] >= d_max:
            break

    complex_points = np.array(complex_points)
    if complex_points.size > 0:
        complex_points = np.stack(
            np.unravel_index(complex_points, stack.shape), axis=1) - pad

    crop = tuple([slice(pad, -pad) for _ in range(kt.ndim)])
    # kt, d, t = (kt == 1).astype(int)[*crop], d[*crop], t[*crop]
    kt, d, t = kt[*crop], d[*crop], t[*crop]
    d[kt != 1] = np.inf
    return complex_points, kt, d, t


if __name__ == "__main__":
    test = sio.loadmat("/Users/ikogbonna/Desktop/Demo.mat")
    IM = test["im"][None]
    SVr = np.concatenate((np.zeros((2, 1)), test["SVr"]), axis=1)

    # --- Run fastmarchingtopo
    nonsimple, KT, D, T = fast_march(IM, SVr, 100, [1, 1, 1])

    peak = np.max(IM)
    for coords in SVr:
        IM[*tuple(np.round(coords).astype(int))] = peak * 2

    plt.imshow(test["im"])
    plt.show()
    plt.imshow(KT[0])
    plt.show()
    plt.imshow((KT[0] == 1).astype(int))
    plt.show()
    plt.imshow(D[0])
    plt.show()
    plt.imshow(T[0])
    plt.show()
