#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 15 2025
@author: ike
"""


# 1. Standard library imports
import warnings

# 3. Local application / relative imports
from .arrays import *
from .slicing import *
from .geometry import *
from .dataframes import *


ERRORS = []

try:
    from .align import *
except ImportError:
    ERRORS.append("align")

if len(ERRORS) > 0:
    warnings.warn(f"cadlabutils.arrays: {', '.join(ERRORS)} not available")
