#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 9 03:38:05 2021
@author: ike
"""


# 1. Standard library imports
import io

# 2. Third-party library imports
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
import seaborn as sns


# set style and colors
sns.set_theme(
    style="ticks", palette="rocket",
    rc={"axes.spines.right": False, "axes.spines.top": False})


def style_ax(
        ax: plt.Axes,
        x_label: str = None,
        y_label: str = None,
        x_ticks: tuple[float, ...] = None,
        y_ticks: tuple[float, ...] = None,
        tick_size: int = 20,
        label_size: int = 20,
        label_weight: str = "bold",
        label_color: str = "black",
        line_width: int = 3
):
    for spine in ax.spines.values():
        spine.set(color=label_color, linewidth=line_width)
    for axis, label, ticks in (
            ("x", x_label, x_ticks), ("y", y_label, y_ticks)):
        if label is not None:
            getattr(ax, f"set_{axis}label")(
                label, fontsize=label_size, fontweight=label_weight,
                color=label_color)
        if ticks is not None:
            getattr(ax, f"set_{axis}lim")(ticks[0], ticks[-1])
            getattr(ax, f"set_{axis}ticks")(ticks)

    # Bold tick labels (can control independently)
    ax.tick_params(
        axis="both", which="major", labelsize=tick_size, width=line_width,
        colors=label_color)
    return ax


def generate_3d_plot(data, save, i_vars, hue):
    c_dict = {l: i for i, l in enumerate(np.unique(data[hue]))}
    groups = data[hue].apply(lambda x: c_dict[x])
    cmap = ListedColormap(
        sns.color_palette(n_colors=np.unique(data[hue]).size).as_hex())
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    sc = ax.scatter(
        *[data[c] for c in i_vars], s=20, c=groups, cmap=cmap, alpha=1)
    ax.set_xlabel(i_vars[0])
    ax.set_ylabel(i_vars[1])
    ax.set_zlabel(i_vars[2])
    plt.legend(
        handles=sc.legend_elements()[0], labels=c_dict.keys(),
        bbox_to_anchor=(1.05, 1), loc=2)
    plt.savefig(save, bbox_inches='tight')
    plt.clf()


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
