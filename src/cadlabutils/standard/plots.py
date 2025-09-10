#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 9 03:38:05 2021

@author: ike
"""


import io
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


sns.set_theme(style="ticks")


def formatTitle(conserved, label: str):
    if conserved in label:
        label = label.replace(conserved, "")

    label = label.strip()
    return label


def getColors(n, palatte="viridis"):
    return sns.color_palette(palatte, n_colors=n)


def getFigAndAxes(rows=1, cols=1, sharex=False, sharey=False, **kwargs):
    plt.rcParams["font.size"] = "15"
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, squeeze=False, figsize=(6 * cols, 4 * rows),
        dpi=150, sharex=sharex, sharey=sharey, **kwargs)
    fig.subplots_adjust(top=0.9)
    return fig, axes


def clearAx(ax):
    ax.axis("off")


def redrawAxis(ax, xaxis=False, yaxis=False, yintercept=0, xintercept=0, lw=2):
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
    (ax.axhline(y=yintercept, xmin=0, xmax=1, lw=lw, color="k")
     if xaxis else None)
    (ax.axvline(x=xintercept, ymin=0, ymax=1, lw=lw, color="k")
     if yaxis else None)


def moveAxis(ax, yintercept=0, xintercept=0, lw=2):
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.spines["bottom"].set_position(('data', yintercept))
    ax.spines["bottom"].set_linewidth(lw)
    ax.spines["left"].set_position(('data', xintercept))
    ax.spines["left"].set_linewidth(lw)


def resetLimits(ax, xs=None, ys=None):
    (ax.set_xlim(*[float(x) for x in xs]) if xs is not None else None)
    (ax.set_ylim(*[float(y) for y in ys]) if ys is not None else None)


def mirrorAxisLimits(ax, axis):
    value = np.abs(ax.get_ylim() if axis == "0" else ax.get_xlim()).max()
    (ax.set_ylim(bottom=-value, top=value) if axis == "0" else ax.set_xlim(
        left=-value, right=value))


def adjustNBins(ax, xbins=10, ybins=8):
    ax.locator_params(axis="x", nbins=xbins)
    ax.locator_params(axis="y", nbins=ybins)


def addHorizontalAxTitle(ax, title):
    ax.set_title(title)


def addVerticalAxTitle(ax, title):
    ax.text(
        0, 0.5, title, verticalalignment='center', rotation=90,
        transform=ax.transAxes)


def addAxisLabels(ax, xlabel=None, ylabel=None):
    (ax.set_xlabel(xlabel) if xlabel is not None else None)
    (ax.set_ylabel(ylabel) if ylabel is not None else None)


def specifyAxisTicks(ax, xticks=None, yticks=None):
    (ax.xaxis.set_ticks(xticks) if xticks is not None else None)
    (ax.yaxis.set_ticks(yticks) if yticks is not None else None)


def removeAxisLabels(ax, xlabel=False, ylabel=False):
    (ax.labels.xaxis.set_visible(False) if xlabel else None)
    (ax.labels.yaxis.set_visible(False) if ylabel else None)


def addLogAxis(ax, xlog=False, ylog=False):
    (ax.set_xscale('log') if xlog else None)
    (ax.set_yscale('log') if ylog else None)


def addLegend(ax):
    ax.legend(loc="center left", frameon=False, bbox_to_anchor=(1, 1))


def shadeVerticalBox(ax, start, stop, alpha=0.05):
    ax.axvspan(start, stop, color="gray", lw=0, alpha=alpha)


def annotatePatches(ax):
    for p in ax.patches:
        num = '{:.1e}'.format(p.get_height())
        num = (num[:3] if "e+00" in num else num)
        ax.annotate(
            num, (p.get_x() + (p.get_width() / 2), 0.9),
            ha='center', va='center')


def linePlot(
        ax, data, yCol=None, xCol=None, hCol=None, hueDict=None, bootN=10,
        color=None, lw=2, raw=False, order=None, **kwargs):
    sns.lineplot(
        x=xCol, y=yCol, hue=hCol, data=data, palette=hueDict, n_boot=bootN,
        ax=ax, color=color, linewidth=lw,
        hue_order=order, **kwargs)
    print("here")
    if raw:
        sns.lineplot(
            x=xCol, y=yCol, hue=hCol, data=data, palette=hueDict,
            n_boot=bootN, ax=ax, color=color, linewidth=lw, estimator=None,
            alpha=0.2, hue_order=order, **kwargs)


def boxPlot(
        ax, data, yCol, cCol=None, hCol=None, hueDict=None, color=None, lw=2,
        raw=False, ori="v", order=None):
    sns.boxplot(
        x=cCol, y=yCol, hue=hCol, data=data, palette=hueDict, ax=ax,
        color=color, orient=ori, linewidth=lw, hue_order=order)
    if raw:
        data = (data.sample(n=50, axis=0) if data.shape[0] > 200 else data)
        sns.swarmplot(
            x=cCol, y=yCol, hue=hCol, data=data, ax=ax, color="k", orient=ori,
            size=3)


def barPlot(
        ax, data, yCol, cCol=None, hCol=None, hueDict=None, ci=95, bootN=1000,
        color=None, lw=2, raw=False, ori="v", order=None):
    sns.barplot(
        x=cCol, y=yCol, hue=hCol, data=data, palette=hueDict, ci=ci,
        n_boot=bootN, ax=ax, color=color, orient=ori, linewidth=lw,
        capsize=.2, hue_order=order, errcolor="k", edgecolor=".2")
    if raw:
        data = (data.sample(n=50, axis=0) if data.shape[0] > 200 else data)
        sns.swarmplot(
            x=cCol, y=yCol, hue=hCol, data=data, ax=ax, color="k", orient=ori,
            size=3)


def heatmap(
        ax, data, vmin, vmax, center, **kwargs):
    sns.heatmap(
        ax=ax, data=data, vmin=vmin, vmax=vmax, center=center, **kwargs)


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


def figToImage(title="", fig=None):
    fig = (plt.gcf() if fig is None else fig)
    plt.tight_layout()
    fig.suptitle(title, y=.95, fontsize=10)
    buffer = io.BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    figure = Image.open(buffer)
    plt.close("all")
    return figure


# def plot_stats(
#         cls,
#         file_csv: str,
#         epochs: int,
#         x_axis: str
# ):
#     """
#     Plot loss and accuracy metrics across training epochs.
#
#     Args:
#         file_csv (str):
#             Path to CSV file containing training statistics.
#         epochs (int):
#             Max number of training epochs.
#         x_axis (str):
#             Variable to plot on x-axis. Options are "epoch" (self._CE) or
#             "samples" (self._CS).
#     """
#     # load statistics
#     stats = pd.read_csv(file_csv, usecols=cls._ALL[:-1]).drop_duplicates()
#
#     # normalize loss to maximum value, pivot to long form
#     # stats[[cls._CL, cls._CLS]] /= stats[cls._CL].max()
#     stats = stats.melt(
#         id_vars=[cls._CN, cls._CS, cls._CE, cls._CM], var_name="stat",
#         value_vars=[cls._CL, cls._CA], value_name="normalized value")
#
#     # set style, colors, and create line plot
#     sns.set_theme(style="ticks", rc={
#         "axes.spines.right": False, "axes.spines.top": False})
#     g = sns.relplot(
#         data=stats, x=x_axis, y="normalized value", row="stat",
#         col=cls._CN, hue=cls._CM, errorbar=("se", 2), kind="line",
#         aspect=2, palette=sns.color_palette("rocket", n_colors=3))
#
#     # clean up and plot
#     x_min = 0 if x_axis == cls._CE else 1
#     x_max = epochs - 1 if x_axis == cls._CE else stats[cls._CS].max()
#     g.set(xlim=(x_min, x_max), ylim=(0, 1))
#     plt.show()
