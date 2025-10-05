#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 15 2025
@author: ike
"""


import warnings

from .arrays import *
from .slicing import *
from .geometry import *
from .dataframe import *


ERRORS = []

try:
    from .align import *
except ImportError:
    ERRORS.append("align")

if len(ERRORS) > 0:
    warnings.warn(f"cadlabutils.arrays: {', '.join(ERRORS)} not available")
