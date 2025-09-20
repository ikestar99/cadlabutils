#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 15 2025
@author: ike
"""


import warnings

from . import csvs as csv


ERRORS = []

try:
    from . import h5s
except ImportError:
    h5s = None
    ERRORS.append("h5s")
try:
    from . import lifs
except ImportError:
    lifs = None
    ERRORS.append("lifs")
try:
    from . import remote
except ImportError:
    remote = None
    ERRORS.append("remote")
try:
    from . import tiffs
except ImportError:
    tiffs = None
    ERRORS.append("tiffs")
try:
    from . import zarrs
except ImportError:
    zarrs = None
    ERRORS.append("zarrs")


if len(ERRORS) > 0:
    warnings.warn(f"cadlabutils.files: {', '.join(ERRORS)} not available")
