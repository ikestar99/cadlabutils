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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
from PIL import Image


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
