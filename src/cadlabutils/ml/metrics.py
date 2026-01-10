#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 09:00:00 2025
@author: Ike
"""


# 2. Third-party library imports
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

# 3. Local application / relative imports
from .utils import forward_pass, set_mode


def count_parameters(
        model: nn.Module
):
    """Count total and trainable parameter in a model.

    Parameters
    ----------
    model : nn.Module
        Model for which to count parameters.

    Returns
    -------
    total : int
        Total number of parameters.
    trainable : int
        Total number of trainable parameters.

    Examples
    --------
    Parameters in simple dense linear model.
    >>> test_model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(5)])
    >>> total_count, trainable_count = count_parameters(test_model)
    >>> total_count
    550
    >>> trainable_count
    550
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def simulate_batch_size(
        model: nn.Module,
        sample: torch.tensor,
        device: torch.device,
        target: torch.tensor = None,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        sample_dtype: torch.dtype = torch.float32,
        target_dtype: torch.dtype = torch.int64,
        start_size: int = 5,
        scalar: float = 0.75
):
    """Simulate optimum batch size that can fit on a given hardware device.

    Parameters
    ----------
    model : nn.Module
        Model with which to perform inference.
    sample : torch.tensor
        Input data on which to run inference. Should not include a batch
        dimension.
    device : torch.device, optional
        Device for which to compute an optimum batch size.
    target : torch.tensor, optional
        Ground truth labels corresponding to the input samples. Should not
        include a batch dimension.
        Defaults to None, in which case loss is not computed.
    criterion : nn.Module, optional
        Instantiated loss function. Passed to forward_pass function.
    optimizer : torch.optim.Optimizer, optional
        Instantiated optimizer. Passed to forward_pass function.

    Returns
    -------
    scaled_max : int
        Scaled peak batch size.

    Other Parameters
    ----------------
    sample_dtype : torch.dtype, optional
        Datatype of input sample. Should match model datatype.
        Defaults to torch.float32, or single-precision.
    target_dtype : torch.dtype, optional
        Datatype of ground truth. Should match loss function.
        Defaults to torch.int64.
    start_size : int, optional
        Known safe batch size with which to begin search.
        Defaults to 1.
    scalar : float, optional
        Fraction of peak size to return as optimum batch size.
        Defaults to 0.75.

    Notes
    -----
    For training loop simulations using BatchNorm layers, `start_size` must be
    >= 2.
    """
    bs_l, bs_h = start_size, start_size
    binary_search = False
    sample = sample.unsqueeze(0)
    target = None if target is None else torch.tensor(target).unsqueeze(0)

    # loop until optimum value found
    set_mode(
        model, train=target is not None, device=device, dtype=sample_dtype)
    while True:
        # generate synthetic batch data
        try_sample = sample.expand(bs_h, *sample.shape[1:])
        try_target = None if target is None else target.expand(
            bs_h, *target.shape[1:])

        # stop if binary search active and no additional increase possible
        delta = (bs_h - bs_l) // 2
        if binary_search and delta <= 1:
            scaled_max = max(1, int(bs_l * scalar))
            return scaled_max

        try:
            # forward pass w/o backpropagation and optimization
            _ = forward_pass(
                model, sample=try_sample, device=device, target=try_target,
                criterion=criterion, optimizer=optimizer,
                sample_dtype=sample_dtype, target_dtype=target_dtype)

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


def classifier_stats(
        output: torch.tensor,
        target: torch.tensor,
        soft_mat: torch.tensor = None,
        hard_mat: torch.tensor = None,
        logits: bool = True
):
    """Compute accuracy metrics for classification model.

    Parameters
    ----------
    output : torch.tensor
        Model prediction. Has shape (batch, classes, ...).
    target : torch.tensor
        Ground truth labels. Has shape (batch, ...) where trailing dimensions
        match those in `output`. Values should be ``int`` dtype.
    soft_mat : torch.tensor, optional
        Running soft confusion matrix of class-wise prediction probabilities.
        Has shape (true classes, predicted classes).
        Defaults to None, in which case confusion matrix is omitted.
    hard_mat : torch.tensor, optional
        Running discrete confusion matrix of class-wise predictions. Same
        shape as `probabilities`.
    logits : bool, optional
        If True, values in `output` are interpreted as raw logits. If False,
        values in `output` are interpreted as probabilities.
        Defaults to True.

    Returns
    -------
    accuracy : float
        Prediction accuracy averaged across batches.
    soft_mat : torch.tensor | None
        `soft_mat` updated with values from current batch.
    hard_mat : torch.tensor | None
        `hard_mat` updated with values from current batch.
    """
    # flatten auxiliary dimensions
    C = output.size(1)
    output = output.permute(0, *range(2, output.ndim), 1).reshape(-1, C)
    target = target.view(-1).long()

    # update probability distribution confusion matrix
    output = F.softmax(output, dim=1) if logits else output
    if soft_mat is not None:
        soft_mat.scatter_add_(0, target[:, None].expand(-1, C), output)

    # compute predicted labels and accuracy
    output = torch.argmax(output, dim=1)
    accuracy = (output == target).float().mean().item()

    # update discrete confusion matrix
    if hard_mat is not None:
        index = target * hard_mat.size(0) + output
        hard_mat += torch.bincount(index, minlength=C ** 2).reshape(C, C)

    return accuracy, soft_mat, hard_mat
