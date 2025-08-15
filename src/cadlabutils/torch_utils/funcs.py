#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 09:00:00 2025
@author: Ike
"""


import torch
import torch.nn as nn

from pathlib import Path
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file, load_file


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
        gpu: int | None = 0
):
    """
    Identify a CUDA-enabled device on which to perform parallelized
    computations. Will use CPU if CUDA and/or GPU are unavailable.

    Args:
        gpu (int | None, optional):
            Index of CUDA device to use. if None, use CPU.
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
        dtype: torch.dtype = torch.float32,
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
        dtype (torch.dtype, optional):
            Datatype of model parameters.
            Defaults to torch.float32, or single-precision.

    Returns:
        model (nn.Module):
            Model with which to perform inference, prepared for specified mode
            and transferred to device with indicated precision.
    """
    model = model.to(device, dtype=dtype)
    model = model.train() if train else model.eval()
    torch.set_grad_enabled(train)
    return model


def forward_pass(
        model: nn.Module,
        sample: torch.tensor,
        device: torch.device,
        target: torch.tensor = None,
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
        device (torch.device, optional):
            Device on which to perform inference.
        target (torch.tensor, optional):
            Ground truth labels corresponding to the input samples.
            Defaults to None, in which case loss is not computed.
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
            Input data on which to run inference. Should not include a batch
            dimension.
        device (torch.device, optional):
            Device for which to compute an optimum batch size.
        target (torch.tensor, optional):
            Ground truth labels corresponding to the input samples. Should not
            include a batch dimension.
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
        (int):
            Scaled peak batch size.
    """
    bs_l, bs_h = start_size, start_size
    binary_search = False
    sample = sample.unsqueeze(0)
    target = target.unsqueeze(0) if target is not None else None
    model = model.to(device, dtype=sample.dtype)

    # loop until optimum value found
    while True:
        # generate synthetic batch data
        try_sample = sample.expand(bs_h, *sample.shape[1:])
        try_target = None if target is None else target.expand(
            bs_h, *target.shape[1:])

        # stop if binary search active and no additional increase possible
        delta = (bs_h - bs_l) // 2
        if binary_search and delta <= 1:
            return max(1, int(bs_l * scalar))

        try:
            # forward pass w/o backpropagation and optimization
            _, _ = forward_pass(
                model, sample=try_sample, device=device, target=try_target,
                criterion=criterion, optimizer=optimizer,
                sample_dtype=sample.dtype,
                target_dtype=target.dtype if target is not None else None)

            # if successful, current high becomes new low
            bs_l = bs_h
            bs_h += delta if binary_search else bs_h

        except RuntimeError as e:
            if "memory" in str(e).lower():
                # if memory error detected, switch to binary search
                binary_search = True
                bs_h -= delta
            else:
                raise e

        finally:
            # clean up GPU memory for next sweep
            del try_sample, try_target
            torch.cuda.empty_cache()


def save(
        file: Path,
        model: nn.Module,
        **kwargs
):
    """
    Save model parameters as a series of checkpoint files.

    Args:
        file (Path):
            location in which to safe checkpoints.
        model (nn.module):
            Model to save in checkpoint file. Model parameters are saved using
            secure .safetensors format.
        **kwargs
            Keyword arguments. Additional items (ex. optimizer state) are saved
            in a tandem .pth file.
    """
    # save model parameters as safetensors format
    save_file(model.state_dict(), file.with_suffix(".safetensors"))

    # save auxiliary values
    if len(kwargs) > 0:
        file_pth = file.with_suffix(".pth")
        saved = torch.load(file_pth) if file_pth.is_file() else {}
        saved = saved.update(kwargs)
        torch.save(saved, file_pth)


def load(
        file: Path,
        model: nn.Module,
        device: torch.device
):
    """
    Load saved model from a series of checkpoint files.

    Args:
        file (Path):
            location in which to safe checkpoints.
        model (nn.module):
            Model to save in checkpoint file. Model parameters are saved using
            secure .safetensors format.
        device (torch.device):
            Device on which to load saved tensors.

    Returns:
        model (nn.Module):
            Model with loaded parameters on specified device.
        extras (dict):
            Additional values stored in companion .pth, mapped to device.
    """
    # load model parameters from safetensors format
    params = load_file(file.with_suffix(".safetensors"))
    model.load_state_dict({k: v.to(device) for k, v in params.items()})

    # load auxiliary values
    file_pth = file.with_suffix(".pth")
    extras = {} if not file_pth.is_file() else torch.load(
        file_pth, map_location=device)
    return model, extras
