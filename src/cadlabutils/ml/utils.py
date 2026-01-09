#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 09:00:00 2025
@author: Ike
"""


# 1. Standard library imports
from pathlib import Path

# 2. Third-party library imports
import torch
import torch.cuda as cuda
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file, save_file


_SAFE = ".safetensors"
_PTH = ".pth"


def is_better(
        loss: tuple[float, float] = None,
        acc: tuple[float, float] = None,
        loss_tol: float = 0.01,
        acc_tol: float = 0.005,
        eps: float = 1e-8
):
    """Determine if current performance is better than prior best performance.

    Parameters
    ----------
    loss : tuple[float, float], optional
        Loss metrics from current and best epochs.
        Defaults to None, in which case loss is not considered.
    acc : float, optional
        Accuracy metrics from current and best epochs
        Defaults to None, in which case accuracy is not considered.
    loss_tol
    acc_tol

    Returns
    -------
    better : bool
        If True, current metrics are better than prior best performance based
        on decreasing loss first, with increasing accuracy as a tie breaker.

    Raises
    ------
    ValueError
        Must provide `loss` or `acc`.

    Notes
    -----
    Any float values for `curr_loss` and `curr_acc` are considered better than
    ``None`` for `best_loss` and `best_acc`.

    Examples
    --------
    Performance is better for current model.
    >>> is_better(loss=(0.5, 0.7))
    True
    >>> is_better(acc=(0.6, 0.3))
    True
    >>> is_better(loss=(0.5, 0.504), acc=(0.62, 0.6))
    True

    Performance is better for prior models.
    >>> is_better(loss=(0.5, 0.3))
    False
    >>> is_better(acc=(0.6, 0.9))
    False
    >>> is_better(loss=(0.5, 0.504), acc=(0.6, 0.6))
    False
    """
    if loss is None and acc is None:
        raise ValueError("Must provide `loss` or `acc`")

    if loss is not None:
        ratio = (loss[0] + eps) / (loss[1] + eps)
        if ratio <= 1 - loss_tol:
            return True
        elif ratio >= 1 + loss_tol:
            return False

    if acc is not None:
        if acc[0] - acc[1] >= acc_tol:
            return True

    return False


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


def get_cuda_memory(
        device: torch.device,
        scale: int = 0
):
    """Report memory consumption on cuda-enabled inference device.

    Parameters
    ----------
    device : torch.device
        CUDA device for which to profile memory usage.
    scale : int, optional
        Scale memory in bytes to a power of 2^10.
        Defaults to 0, in which case returned memories are in bytes.

    Returns
    -------
    allocated : int
        Active memory pool.
    reserved : int
        Reserved memory pool.
    total : int
        Total memory pool.

    Raises
    ------
    ValueError
        Passed device is not cuda enabled.
    """
    scalar = 1024 ** scale
    if device.type == "cpu" or not torch.cuda.is_available():
        raise ValueError("Passed device is not cuda enabled.")

    allocated = torch.cuda.memory_allocated(device) / scalar
    reserved = torch.cuda.memory_reserved(device) / scalar
    total = torch.cuda.get_device_properties(device).total_memory / scalar
    return allocated, reserved, total


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


def get_dataloader(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        workers: int = 4,
        drop_last: bool = False,
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
    drop_last : bool, optional
        If True and `batch_size` is 1, drop final batch if number of samples
        is less than `batch_size`.
        Defaults to True.

    Returns
    -------
    loader : DataLoader
        Instantiated DataLoader.

    Notes
    -----
    If `dataset` is not divisible into equal batches of `batch_size` samples,
    The final non-`batch_size` batch will be discarded if it includes a single
    sample and `batch_size` > 1.
    """
    remainder = len(dataset) % batch_size
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=True, persistent_workers=True,
        drop_last=all((remainder > 0, batch_size != 1, drop_last)), **kwargs)
    return loader


def save(
        file: Path,
        model: nn.Module,
        save_dict: dict[str, object] = None,
        **kwargs
):
    """Save model parameters as series of checkpoint files.

    Parameters
    ----------
    file : Path
        location in which to save checkpoints.
    model : nn.module
        Model to save in checkpoint file.
    save_dict : dict[str, object], optional
        key : str
            Key of state dict to save in companion .pth file.
        value : object
            Object to save state via `state_dict` method.
        Default to None.
    **kwargs
        Keyword arguments. Additional items are saved in a tandem .pth file.

    See Also
    --------
    load : Complementary load function.

    Notes
    -----
    Saving a model yields at least one file, a .safetensors file that stores
    model parameters in a secure pickle-independent format. Any additional
    values are saved as a dictionary in a pytorch .pth file. If this secondary
    file already exists, key: value pairs will be overwritten if they match
    keys specified in `save_dict` or `kwargs`.

    Examples
    --------
    Save model with additional mid-training data.
    >> test_model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(5)])
    >> test_path = Path("~/Desktop/test_save")
    >> save(test_path, test_model, epoch=10)
    """
    # save model parameters as safetensors format
    save_file(model.state_dict(), file.with_suffix(_SAFE))
    save_dict = {} if save_dict is None else save_dict

    # save auxiliary values
    if len(save_dict) + len(kwargs) > 0:
        file_pth = file.with_suffix(_PTH)
        saved = torch.load(file_pth) if file_pth.is_file() else {}
        saved.update(kwargs)
        saved.update({k: v.state_dict() for k, v in save_dict.items()})
        torch.save(saved, file_pth)


def load(
        file: Path,
        model: nn.Module,
        device: torch.device,
        load_dict: dict[str, object] = None
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
    load_dict : dict[str, object], optional
        key : str
            Key of state dict stored in companion .pth file.
        value : object
            Object to update in place using `load_state_dict` method.
        Defaults to None.

    Returns
    -------
    extras : dict[str: Any]
        Additional values stored in companion .pth, mapped to device.

    See Also
    --------
    save : Complementary save function.

    Notes
    -----
    For keys specified in `load_dict`, the corresponding value will be updated
    The structure of `extras` depends on the way in which `model` was initially
    saved at `file`. If only model parameters were saved, `extras is an empty
    dictionary. Otherwise, `extras` is a dictionary of `**kwargs` supplied when
    model was initially saved, excluding keys in `load_dict`.

    For keys in `load_dict`, the corresponding value will be updated with a
    state dict at the same key in saved companion file.

    Examples
    --------
    >> test_model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(5)])
    >> test_path = Path("~/Desktop/test_save")
    >> test_extras = load(test_path, test_model, device=get_device(None))
    >> test_extras
    {'epoch': 10}
    """
    # load model parameters from safetensors format
    params = load_file(file.with_suffix(_SAFE))
    model.load_state_dict({k: v.to(device) for k, v in params.items()})

    # load auxiliary values
    file_pth = file.with_suffix(_PTH)
    extras = {} if not file_pth.is_file() else torch.load(
        file_pth, map_location=device)

    # load state dicts
    load_dict = {} if load_dict is None else load_dict
    for k in set(load_dict.keys()).intersection(extras.keys()):
        load_dict[k].load_state_dict(extras[k])

    # isolate remaining values
    extras = {k: v for k, v in extras.items() if k not in load_dict}
    return extras


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

    Notes
    -----
    Adjusts model in-place.
    """
    model.to(device, dtype=dtype)
    _ = model.train() if train else model.eval()
    torch.set_grad_enabled(train)


def forward_pass(
        model: nn.Module,
        sample: torch.tensor,
        device: torch.device,
        target: torch.tensor = None,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
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
    optimizer : torch.optim.Optimizer, optional
        Instantiated optimizer used for model parameter optimization after
        backpropagation.
        Defaults to None, in which case gradients are unaltered.

    Returns
    -------
    output : torch.tensor
        Output of forward pass through model.
    loss : torch.tensor | None
        Output of criterion. None if loss is not computed.
    target : torch.tensor | None
        Ground truth labels moved to corresponding device. None if `target` is
        None.

    Other Parameters
    ----------------
    sample_dtype : torch.dtype, optional
        Datatype of input sample. Should match model datatype.
        Defaults to torch.float32, or single-precision.
    target_dtype : torch.dtype, optional
        Datatype of ground truth. Should match loss function.
        Defaults to torch.int64.

    Notes
    -----
    Simple forward pass. Requires `sample` and `target` to be ``torch.tensor``
    where `model(sample)` and `criterion(model(sample), target)` are both
    syntactically valid.
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

    return output, loss, target
