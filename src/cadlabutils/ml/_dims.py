#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 07 09:00:00 2026
@author: ike
"""


# 2. Third-party library imports
import torch.nn as nn


class _D2:
    _D_I = 2
    _NORM = "2d"
    _CONV = nn.Conv2d
    _POOL = nn.MaxPool2d
    _TPOS = nn.ConvTranspose2d
    _AVGP = nn.AdaptiveAvgPool2d


class _D3:
    _D_I = 3
    _NORM = "3d"
    _CONV = nn.Conv3d
    _POOL = nn.MaxPool3d
    _TPOS = nn.ConvTranspose3d
    _AVGP = nn.AdaptiveAvgPool3d
