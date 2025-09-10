#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 09:00:00 2025
@author: Ike
"""


import torch
import torch.nn as nn
import torch.cuda as cuda

from pathlib import Path
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file, load_file


def get_device(
        gpu: int | None = 0
):
    """Identify a CUDA-enabled device on which to perform computation.

    Parameters
    ----------
    gpu : int | None, optional
        Index of CUDA device to use. if None, use CPU.
        Defaults to 0.

    Returns
    -------
    device : torch.device
        Device on which to perform computations.

    Notes
    -----
    Will use CPU if CUDA and/or GPU are unavailable.

    Examples
    --------
    Get cpu device.
    >>> get_device(None)
    device(type='cpu')
    """
    device = torch.device(
        f"cuda:{min(gpu, cuda.device_count() - 1)}"
        if cuda.is_available() and gpu is not None else "cpu")
    return device


def get_device_names(
):
    """Get the names and sizes of available CUDA devices.

    Returns
    -------
    device_info : dict[int, dict[str, int | float]]
        key : int
            Device index.
        value : dict[str, str | float]
            Contains two keys:
            -   "name": device name, str.
            -   "GiBs": device VRAM capacity in GiBs, float.
    """
    device_info = {
        i: {
            "name": cuda.get_device_name(i),
            "GiBs": cuda.get_device_properties(i).total_memory / (1024 ** 3)
        } for i in range(cuda.device_count())} if cuda.is_available() else {}
    return device_info


def get_loader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        workers: int = 4,
        **kwargs
):
    """Create a DataLoader from a Dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset to wrap in a DataLoader.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        If True, shuffle dataset indices before forming batches.
        Defaults to True.
    workers : int, optional
        Number of parallel workers.
        Defaults to 4.

    Returns
    -------
    loader : DataLoader
        Instantiated DataLoader.
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=True, persistent_workers=True, **kwargs)
    return loader


def save(
        file: Path,
        model: nn.Module,
        **kwargs
):
    """Save model parameters as series of checkpoint files.

    Parameters
    ----------
    file : Path
        location in which to safe checkpoints.
    model : nn.module
        Model to save in checkpoint file.
    **kwargs
        Keyword arguments. Additional items are saved in a tandem .pth file.

    See Also
    --------
    cadlabutils.learning.funcs.load : Complementary load function.

    Notes
    -----
    Saving a model yields at least one file, a .safetensors file that stores
    model parameters in a secure pickle-independent format. Any additional
    values are saved as a dictionary in a pytorch .pth file. If this secondary
    file already exists, key: value pairs will be overwritten if that match
    keys specified in `kwargs`. Remaining key: value pairs are preserved.

    Examples
    --------
    Save model with additional mid-training data.
    >> test_model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(5)])
    >> test_path = Path("~/Desktop/test_save")
    >> save(test_path, test_model, epoch=10)
    """
    # save model parameters as safetensors format
    save_file(model.state_dict(), file.with_suffix(".safetensors"))

    # save auxiliary values
    if len(kwargs) > 0:
        file_pth = file.with_suffix(".pth")
        saved = torch.load(file_pth) if file_pth.is_file() else {}
        saved.update(kwargs)
        torch.save(saved, file_pth)


def load(
        file: Path,
        model: nn.Module,
        device: torch.device
):
    """
    Load saved model from a series of checkpoint files.

    Parameters
    ----------
    file : Path
        Path to saved model parameters (.safetensors).
    model : nn.module
        Model in which to load parameters.
    device : torch.device
        Device on which to load saved tensors.

    Returns
    -------
    model : nn.Module
        Model with loaded parameters on specified device.
    extras : dict[str: Any]
        Additional values stored in companion .pth, mapped to device.

    See Also
    --------
    cadlabutils.learning.funcs.save : Complementary save function.

    Notes
    -----
    The structure of `extras` depends on the way in which `model` was initially
    saved at `file`. If only model parameters were saved, `extras is an empty
    dictionary. Otherwise, `extras` is a dictionary of `**kwargs` supplied when
    model was initially saved.

    Examples
    --------
    >> test_model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(5)])
    >> test_path = Path("~/Desktop/test_save")
    >> test_model, test_extras = load(
    ..     test_path, test_model, device=get_device(None))
    >> test_extras
    {'epoch': 10}
    """
    # load model parameters from safetensors format
    params = load_file(file.with_suffix(".safetensors"))
    model.load_state_dict({k: v.to(device) for k, v in params.items()})

    # load auxiliary values
    file_pth = file.with_suffix(".pth")
    extras = {} if not file_pth.is_file() else torch.load(
        file_pth, map_location=device)
    return model, extras


def set_mode(
        model: nn.Module,
        train: bool,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
):
    """Prepare model for either train or test passes.

    Parameters
    ----------
    model : nn.Module
        Model with which to perform inference.
    train : bool
        If True, enable gradients and computation graph.
    device : torch.device
        Device on which to perform computations.
    dtype : torch.dtype, optional
        Datatype of model parameters.
        Defaults to torch.float32, or single-precision.

    Returns
    -------
    model : nn.Module
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
    """Run inference and compute loss if relevant.

    Parameters
    ----------
    model : nn.Module
        Model with which to perform inference.
    sample : torch.tensor
        Input data on which to run inference.
    device : torch.device, optional
        Device on which to perform inference.
    target : torch.tensor, optional
        Ground truth labels corresponding to the input samples.
        Defaults to None, in which case loss is not computed.
    criterion : nn.Module, optional
        Instantiated loss function to use for backpropagation.
        Defaults to None, in which case loss is not computed.
    optimizer : Optimizer, optional
        Instantiated optimizer used for model parameter optimization after
        backpropagation.
        Defaults to None, in which case gradients are unaltered.

    Returns
    -------
    output : torch.tensor
        Output of forward pass through model.
    target : torch.tensor | None
        Ground truth labels transferred to inference device. None if loss not
        computed.
    loss : torch.tensor | None
        Output of criterion. None if loss is not computed.

    Other Parameters
    ----------------
    sample_dtype : torch.dtype, optional
        Datatype of input sample. Should match model datatype.
        Defaults to torch.float32, or single-precision.
    target_dtype : torch.dtype, optional
        Datatype of ground truth. Should match loss function.
        Defaults to torch.int64.
    """
    loss = None
    if None not in (target, criterion, optimizer):
        # zero gradients
        optimizer.zero_grad()

    # inference with model
    output = model(sample.to(device, non_blocking=True, dtype=sample_dtype))
    if None not in (target, criterion):
        # compute loss
        target = target.to(device, non_blocking=True, dtype=target_dtype)
        loss = criterion(output, target)
        if optimizer is not None:
            # backpropagation, clip gradients, optimize
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

    return output, target, loss
