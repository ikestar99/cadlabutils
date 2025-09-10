#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 09:00:00 2025
@author: Ike
"""


import torch
import torch.nn as nn

from torch.optim import Optimizer
import torch.nn.functional as F

from .utils import forward_pass


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
        optimizer: Optimizer = None,
        start_size: int = 1,
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
    optimizer : Optimizer, optional
        Instantiated optimizer. Passed to forward_pass function.

    Returns
    -------
    scaled_max : int
        Scaled peak batch size.

    Other Parameters
    ----------------
    start_size : int, optional
        Known safe batch size with which to begin search.
        Defaults to 1.
    scalar : float, optional
        Fraction of peak size to return as optimum batch size.
        Defaults to 0.75.
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
            scaled_max = max(1, int(bs_l * scalar))
            return scaled_max

        try:
            # forward pass w/o backpropagation and optimization
            _ = forward_pass(
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


def classification_accuracy(
        output: torch.tensor,
        target: torch.tensor,
        matrix: torch.tensor = None,
        logits: bool = True
):
    """Compute accuracy metrics for classification model.

    Parameters
    ----------
    output : torch.tensor
        Model prediction. Has shape (batch, classes, ...).
    target : torch.tensor
        Ground truth labels. Has shape (batch, ...) where trailing dimensions
        match those in `output`. Valuses should be ``int`` dtype.
    matrix : torch.tensor, optional
        Running confusion matrix of class-wise prediction probabilities. Has
        shape (true classes, predicted classes).
        Defaults to None, in which case confusion matrix is omitted.
    logits : bool, optional
        If True, values in `output` are interpreted as raw logits. If False,
        values in `output` are interpreted as probabilities.
        Defaults toTrue.

    Returns
    -------
    accuracy : float
        Prediction accuracy averaged across batches.
    matrix : torch.tensor | None
        Aggregate confusion matrix of class-wise prediction probabilities.
    """
    # convert logits into probabilities and calculate accuracy
    output = F.softmax(output, dim=1) if logits else output
    accuracy = int(
        (torch.argmax(output, dim=1) == target).sum().item()) / target.numel()

    # update confusion matrix probability distributions per class
    if matrix is not None:
        target = target.reshape(-1)
        output = torch.movedim(output, 1, -1).reshape(-1, output.size(1))
        for c in range(matrix.size(0)):
            matrix[c] += output[target == c].sum(dim=0)

    return accuracy, matrix
