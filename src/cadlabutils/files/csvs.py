#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


from pathlib import Path


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
