#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 9 03:38:05 2021
@author: ike
"""


# 1. Standard library imports
import io

# 2. Third-party library imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.neighbors import KernelDensity


# set style and colors
sns.set_theme(
    style="ticks", palette="rocket",
    rc={"axes.spines.right": False, "axes.spines.top": False})
_SAVE_KWARGS = {"dpi": 300, "bbox_inches": "tight", "pad_inches": 0}


def style_ax(
        ax: plt.Axes,
        x_label: str = None,
        y_label: str = None,
        x_ticks: tuple[float, ...] = None,
        y_ticks: tuple[float, ...] = None,
        x_color: str = None,
        y_color: str = None,
        x_cross: float = 0,
        y_cross: float = 0,
        tick_size: int = 30,
        label_size: int = 25,
        label_weight: str = "bold",
        label_color: str = "black",
        line_width: int = 5,
        draw_yx: bool = False,
):
    for spine in ax.spines.values():
        spine.set(color=label_color, linewidth=line_width)
    for axis, label, ticks, color, cross in (
            ("x", x_label, x_ticks, x_color, y_cross),
            ("y", y_label, y_ticks, y_color, x_cross)):
        if label is not None:
            getattr(ax, f"set_{axis}label")(
                label, fontsize=label_size, fontweight=label_weight,
                color=label_color)
        if ticks is not None:
            getattr(ax, f"set_{axis}lim")(ticks[0], ticks[-1])
            getattr(ax, f"set_{axis}ticks")(ticks)
            bounds = getattr(ax, f"get_{axis}lim")()
            if bounds[0] < cross < bounds[1]:
                l, s = ("h", "bottom") if axis == "y" else ("v", "left")
                getattr(ax, f"ax{l}line")(
                    cross, color="black", linewidth=line_width, zorder=0)
                ax.spines[s].set_visible(False)
        if color is not None:
            getattr(ax, f"{axis}axis").label.set_color(color)

    if draw_yx:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        lo, hi = max(xmin, ymin), min(xmax, ymax)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", zorder=0)
    if x_ticks is not None and x_ticks == y_ticks:
        ax.set_aspect('equal', adjustable='box')

    # Bold tick labels (can control independently)
    ax.tick_params(
        axis="both", which="major", labelsize=tick_size, width=line_width,
        colors=label_color)
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
        clip: tuple,
        bins: int = 100,
        lw: float = 5,
        spacing: float = 0.8
):
    lines = {}
    for h, h_df in data.groupby(hue):
        _y, _x = np.histogram(h_df[x], bins=bins)
        # _x = np.linspace(clip[0], clip[1], bins)
        # _y = h_df[x].dropna().to_numpy()
        # kde = KernelDensity(
        #     kernel="gaussian", bandwidth=0.2).fit(_y[:, None])
        # _y = np.exp(kde.score_samples(_x[:, None]))
        lines[h] = _x[:-1], _y / np.trapezoid(_y, _x[:-1])

    offset = max([lines[k][1].max() for k in lines]) * spacing
    peak = 0
    for i, key in enumerate(hue_order):
        if key not in lines:
            continue

        _x, _y = lines[key]
        _dy = offset * (len(hue_order) - 1 - i)
        ax.plot(_x, _y + _dy, color=palette[key], linewidth=lw)
        ax.fill_between(_x, _y + _dy, _dy, color=palette[key], alpha=0.2)
        ax.plot(_x, [_dy] * len(_x), color="gray", linewidth=lw)
        peak = max(peak, np.max(_y + _dy))

    return ax, peak * 1.1


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
