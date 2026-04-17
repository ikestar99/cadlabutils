#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 8 09:00:00 2025
@author: Ike
"""


# 1. Standard library imports
from pathlib import Path

# 2. Third-party library imports
import imagej


_MACROS = {
    f.stem: f for f in
    Path(__file__).parents[3].joinpath("external_macros").glob("*.ijm")}
_FIJI = None


def _get_fiji(
        ij_path: Path
):
    global _FIJI
    if _FIJI is None:
        if ij_path is None:
            raise ValueError("Missing Fiji installation path")

        _FIJI = imagej.init(str(ij_path), mode="headless")

    return _FIJI


def rolling_ball_background(
        tif_path: Path,
        out_path: Path,
        radius: int,
        background: str = "light",
        ij_path: Path = None
):
    _get_fiji(ij_path).py.run_macro(
        _MACROS["fiji_rolling_ball"].read_text(), {
            "inputPath": tif_path, "outputPath": out_path, "radius": radius,
            "background": background})
