#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 09:00:00 2025
@author: Ike
"""


import torch
import torch.nn as nn

from torch.optim import Optimizer
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


def get_loader(
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


def set_mode(
        model: nn.Module,
        train: bool,
        device: torch.device,
        model_dtype: torch.dtype = torch.float32,
):
    """
    Prepare model for either train or test passes.

    Args:
        model (nn.Module):
            Model with which to perform inference.
        train (bool):
            If True, prepare model for backpropagation by enabling gradients.
            If False, Prepare for inference by disabling gradients and
            computation graph.
        device (torch.device):
            Device on which to perform computations.
        model_dtype (torch.dtype, optional):
            Datatype of model parameters.
            Defaults to torch.float32, or single-precision.

    Returns:
        model (nn.Module):
            Model with which to perform inference, prepared for specified mode
            and transferred to device with indicated precision.
    """
    model = model.to(device, dtype=model_dtype)
    model = model.train() if train else model.eval()
    torch.set_grad_enabled(train)
    return model


def forward_pass(
        model: nn.Module,
        sample: torch.tensor,
        target: torch.tensor = None,
        device: torch.device = torch.device("cpu"),
        criterion: nn.Module = None,
        optimizer: Optimizer = None,
        sample_dtype: torch.dtype = torch.float32,
        target_dtype: torch.dtype = torch.int64
):
    """
    Run inference and compute loss if target and loss function are available.

    Args:
        model (nn.Module):
            Model with which to perform inference.
        sample (torch.tensor):
            Input data on which to run inference.
        target (torch.tensor, optional):
            Ground truth labels corresponding to the input samples.
            Defaults to None, in which case loss is not computed.
        device (torch.device, optional):
            Device on which to perform inference.
            Defaults to torch.device("cpu").
        criterion (nn.Module, optional):
            Instantiated loss function to use for backpropagation.
            Defaults to None, in which case loss is not computed.
        optimizer (Optimizer, optional):
            Instantiated optimizer used for model parameter optimization after
            backpropagation.
            Defaults to None, in which case gradients are unaltered.
        sample_dtype (torch.dtype, optional):
            Datatype of input sample. Should match model datatype.
            Defaults to torch.float32, or single-precision.
        target_dtype (torch.dtype, optional):
            Datatype of input sample. Should match loss function.
            Defaults to torch.int64.

    Returns:
        output (torch.tensor):
            Output of forward pass through model.
        loss (torch.tensor | None):
            Output of criterion. Value is None if loss is not computed.
    """
    loss = None
    if None not in (target, criterion, optimizer):
        # zero gradients
        optimizer.zero_grad()

    # inference with model
    output = model(sample.to(device, non_blocking=True, dtype=sample_dtype))
    if None not in (target, criterion):
        # compute loss
        loss = criterion(
            output, target.to(device, non_blocking=True, dtype=target_dtype))
        if optimizer is not None:
            # backpropagation, clip gradients, optimize
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

    return output, loss


def simulate_batch_size(
        model: nn.Module,
        sample: torch.tensor,
        device: torch.device,
        target: torch.tensor = None,
        criterion: nn.Module = None,
        optimizer: Optimizer = None,
        start_size: int = 1,
        scalar: float = 0.75
):
    """
    Use forward pass to simulate the optimum batch size that can fit on a given
    hardware device. Optimum size is a set percentage of the maximum size that
    will fit in memory without raising a RunetimeError or OutOfMemoryError.

    Args:
        model (nn.Module):
            Model with which to perform inference.
        sample (torch.tensor):
            Input data on which to run inference.
        device (torch.device, optional):
            Device for which to compute an optimum batch size.
            Defaults to torch.device("cpu").
        target (torch.tensor, optional):
            Ground truth labels corresponding to the input samples.
            Defaults to None, in which case loss is not computed.
        criterion (nn.Module, optional):
            Instantiated loss function. Passed to forward_pass function.
        optimizer (Optimizer, optional):
            Instantiated optimizer. Passed to forward_pass function.
        start_size (int, optional):
            Known safe batch size with which to begin search.
            Defaults to 1.
        scalar (float, optional):
            Fraction of peak size to return as optimum batch size.
            Defaults to 0.75.

    Returns:
        optimum (int):
            Optimum batch size.
    """
    sizes = [start_size]
    binary_search = False
    sample = sample.unsqueeze(0)
    target = None if target is None else target.unsqueeze(0)
    model = model.to(device, dtype=sample.dtype)
    while True:
        try_sample = sample.repeat(*(sizes[-1:] + [1] * sample.dim()))
        try_target = None if target is None else target.repeat(
            *(sizes[-1:] + [1] * target.dim()))
        try:
            forward_pass(
                model, sample=try_sample, target=try_target, device=device,
                criterion=criterion, optimizer=optimizer,
                sample_dtype=sample.dtype,
                target_dtype=target.dtype if target is not None else None)
            sizes += [sizes[-1] * 2 ]
            max_bs = bs
            bs *= 2  # try bigger
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                break
            else:
                raise e

    # Binary search refinement
    low, high = max_bs, bs
    while low + 1 < high:
        mid = (low + high) // 2
        try:
            torch.cuda.empty_cache()
            optimizer = torch.optim.Adam(model.parameters())
            x = torch.randn((mid, 1, *voxel), device=device).double()
            target = torch.randint(0, len(balance), (mid,), device=device).long()

            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

            low = mid
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                high = mid
            else:
                raise e

    return low
