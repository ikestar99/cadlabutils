#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


# 1. Standard library imports
from pathlib import Path

# 2. Third-party library imports
import pandas as pd


def find_first_row(
        file: Path | str
):
    """Find first numeric row in a csv-like file.

    Parameters
    ----------
    file : Path | str
        If a Path, path must point to a csv-like file (.csv, .swc, ...).
        Otherwise, contains entire contents of csv-like file as a string.

    Returns
    -------
    int | None
        Number of non-numeric rows at beginning of file. If all rows are
        non-numeric, return value is ``None``.
    """
    def _check_rows(
            iterable
    ):
        for i, line in enumerate(iterable):
            try:  # check if the first element is numeric
                float(" ".join(line.strip().split(",")).split(" ")[0])
                return i
            except ValueError:  # if not numeric, check the next row
                pass

    # open the swc file
    if isinstance(file, Path):
        with open(file, 'r') as f:
            return _check_rows(f)

    return _check_rows(file.splitlines())


def append_data(
        file: Path,
        data: pd.DataFrame,
        index: bool = False,
):
    """Either create or append data to a csv file.

    Parameters
    ----------
    file : Path
        Path to save data (csv).
    data : pd.DataFrame
        Data to append.
    index : bool, optional
        If True, include index of `data` in `file`.
        Defaults to ``False``.
    """
    exists = file.is_file()
    data.to_csv(
        file, index=index, mode="a" if exists else "w", header=not exists)
