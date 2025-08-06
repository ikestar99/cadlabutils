#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 09:00:00 2025
@author: Ike
"""


import torch


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable