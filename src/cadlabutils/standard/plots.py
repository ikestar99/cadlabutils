#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 9 03:38:05 2021
@author: ike
"""


# 1. Standard library imports
import io
from pathlib import Path

# 2. Third-party library imports
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree


# set style and colors
sns.set_theme(
    style="ticks", palette="rocket",
    rc={
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.facecolor": (0, 0, 0, 0)})
_SAVE_KWARGS = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0}


def style_ax(
        ax: plt.Axes,
        title: str = None,
        tick_size: int = 15,
        label_size: int = 15,
        label_weight: str = "bold",
        label_color: str = "black",
        line_width: int = 2.5,
        draw_yx: bool = False,
        aspect: tuple = None,
        **kwargs
):
    ax.tick_params(
        axis="both", which="major", labelsize=tick_size, width=line_width)
    for spine in ax.spines.values():
        spine.set(color=label_color, linewidth=line_width)

    if title is not None:
        ax.set_title(title)

    for axis in ("z", "y", "x"):
        side = "left" if axis == "y" else "bottom"
        _k = {
            k.split("_", maxsplit=1)[1]: v for k, v in kwargs.items()
            if v is not None and k.lower().startswith(f"{axis}_")}
        if "label" in _k:
            getattr(ax, f"set_{axis}label")(
                _k["label"], fontsize=label_size, fontweight=label_weight,
                color=_k["color"] if "color" in _k else label_color)
        if "lim" in _k:
            getattr(ax, f"set_{axis}lim")(_k["lim"][0], _k["lim"][-1])
        if "ticks" in _k:
            if "lim" not in _k:
                getattr(ax, f"set_{axis}lim")(_k["ticks"][0], _k["ticks"][-1])

            getattr(ax, f"set_{axis}ticks")(_k["ticks"])
        if "ticklabels" in _k:
            getattr(ax, f"set_{axis}ticklabels")(_k["ticklabels"])
        if "cross" in _k and axis != "z":
            ax.spines[side].set_position(('data', _k["cross"]))
        if "visible" in _k and not _k["visible"] and axis != "z":
            getattr(ax, f"{axis}axis").set_visible(False)
            ax.spines[side].set_visible(False)

    if draw_yx:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        lo, hi = max(xmin, ymin), min(xmax, ymax)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", zorder=0)

    if aspect is not None:
        ax.set_box_aspect(aspect)

    return ax


def save_fig(
        name,
        root_dir,
        fig=None
):
    fig = plt.gcf() if fig is None else fig
    root_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(root_dir / f"{name}.png", **_SAVE_KWARGS)
    plt.close(fig)


def ridgeplot(
        data: pd.DataFrame,
        x: str,
        ax,
        hue: str,
        hue_order: list,
        palette: dict,
        clip: tuple = None,
        bins: int = 100,
        lw: float = 5,
        spacing: float = 0.7,
        padding: float = 0.1,
):
    lines = {}
    clip = np.array([data[x].min(), data[x].max()] if clip is None else clip)
    _x = np.linspace(
        *(clip + (np.ptp(clip) * np.array([-padding, padding]))), bins)
    for h, h_df in data.groupby(hue):
        lines[h] = gaussian_kde(h_df[x].dropna())(_x)

    offset = max([lines[k].max() for k in lines]) * spacing
    peak = 0
    for i, key in enumerate(hue_order):
        if key not in lines:
            continue

        _y = lines[key]
        _dy = offset * (len(hue_order) - 1 - i)
        ax.plot(_x, _y + _dy, color="w", linewidth=5, zorder=i + 1)
        ax.fill_between(_x, _y + _dy, _dy, color=palette[key], zorder=i + 1)
        ax.plot(_x, [_dy] * len(_x), color="black", linewidth=2, zorder=i + 1)
        peak = max(peak, np.max(_y + _dy))

        r_just = _x[np.argmax(_y)] <= np.sum(clip) / 2
        ax.text(
            x=clip[int(r_just)], y=_dy + (offset * 0.5), s=key,
            color=palette[key], fontsize=25, ha="right" if r_just else "left",
            va="center")

    return ax, peak, (min(_x), max(_x))


def fig_to_im(
        fig: plt.Figure
):
    """Convert matplotlib figure to a PIL Image.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to convert.

    Returns
    -------
    im : PIL Image
        `fig` converted into a PIL Image.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    im = Image.open(buffer)
    plt.close(fig)
    return im


def rotate_3d(
        fig: plt.Figure,
        name: str,
        root_dir: Path,
        elev: float = 25,
        seconds: float = 5,
        fps: int = 20
):
    def _rotate(idx):
        azimuth = (360 * idx / (fps * seconds)) - 180
        for ax in [a for a in fig.axes if a.name == "3d"]:
            ax.view_init(elev=elev, azim=azimuth)

    root_dir.mkdir(exist_ok=True, parents=True)
    anim = FuncAnimation(
        fig, _rotate, frames=int(fps * seconds), blit=False, repeat=False)
    anim.save(root_dir / f"{name}.gif", writer=PillowWriter(fps=fps))


def surfaceplot(
        data: np.ndarray,
        ax: plt.Axes,
        z: np.ndarray = None,
        n: int = 150,
        k: int = 200,
        x_lim: tuple = None,
        y_lim: tuple = None,
        **kwargs
):
    # KD-tree over the actual observations
    x_lim = x_lim if x_lim is not None else (data[0].min(), data[0].max())
    y_lim = y_lim if y_lim is not None else (data[1].min(), data[1].max())
    z = z if z is not None else data[:, -1]
    x, y = np.meshgrid(
        np.linspace(*x_lim, n), np.linspace(*y_lim, n), indexing="ij")

    # Find k nearest observations at each grid location
    _, idx = cKDTree(data[:, :2]).query(
        np.column_stack([x.ravel(), y.ravel()]), k=k)
    surface = ax.plot_surface(
        x, y, z[idx].mean(axis=1).reshape(x.shape), cmap="rocket", linewidth=0,
        antialiased=True, alpha=0.5, **kwargs)
    return ax, surface
