#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 09:00:00 2025
@author: Ike
"""


import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


"""
Utility functions for Pytorch models and training workflows.
"""


def count_parameters(
        model: nn.Module
):
    """
    Identify total and trainable parameter counts for a model.

    Args:
        model (nn.Module):
            Model for which to count parameters.

    Returns:
        total (int):
            Total number of parameters.
        trainable (int):
            Total number of trainable parameters.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_device(
        gpu: int = 0
):
    """
    Identify a CUDA-enabled device on which to perform parallelized
    computations. Will use CPU if CUDA and/or GPU are unavailable.

    Args:
        gpu (int, optional):
            Index of CUDA device to use.
            Defaults to 0.

    Returns:
        device (torch.device):
            Device on which to perform computations.
    """
    device = torch.device(
        f"cuda:{min(gpu, torch.cuda.device_count() - 1)}"
        if torch.cuda.is_available() and gpu is not None else "cpu")
    return device


def _get_loader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        workers: int = 4,
        **kwargs
):
    """
    Create a DataLoader from a Dataset.

    Args:
        dataset (Dataset):
            Dataset to wrap in a DataLoader.
        batch_size (int):
            Number of samples per batch.
        shuffle (bool):
            If True, shuffle dataset indices before forming batches.
            Defaults to True.
        workers (int, optional):
            Number of parallel workers.
            Defaults to 4.

    Returns:
        loader (DataLoader):
            Instantiated DataLoader.
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=True, persistent_workers=True, **kwargs)
    return loader
