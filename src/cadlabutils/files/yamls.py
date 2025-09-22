#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 15 2025
@author: ike
"""


import yaml

from pathlib import Path


def to_yaml(
        file: Path,
        data: dict
):
    """Save data in yaml format.

    Parameters
    ----------
    file : Path
        Path to save data (.yaml).
    data : dict
        Data to save.
    """
    with open(file.with_suffix(".yaml"), "w") as f:
        yaml.safe_dump(data, f)


def from_yaml(
        file: Path
):
    """Load data in yaml format.

    Parameters
    ----------
    file : Path
        Path to load data (.yaml).

    Returns
    -------
    data : dict
        Saved data.
    """
    with open(file.with_suffix(".yaml"), "r") as f:
        data = yaml.safe_load(f)
        return data
