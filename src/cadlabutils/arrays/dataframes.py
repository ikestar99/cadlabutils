#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import pandas as pd


def format_str_index(
        index,
        replace: str = '_'
):
    """Format string index values to uppercase alphanumeric characters.

    Parameters
    ----------
    index : pd.Index | pd.Series
        Index of labels to format as strings.
    replace : str, optional
        Character used to replace non-alphanumeric characters in `index`.
        Defaults to '_'.

    Returns
    -------
    formatted_index : pd.Series
        `index` with non-alphanumeric characters replaced with `replace` and
        other characters converted to uppercase.
    """
    formatted_index = index.to_series().str.replace(
        r'[^0-9A-Za-z]', replace, regex=True).str.upper()
    return formatted_index
