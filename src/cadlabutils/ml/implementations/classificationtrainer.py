#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


# 2. Third-party library imports
import numpy as np
import torch

# 3. Local application / relative imports
import cadlabutils as cdu
import cadlabutils.files as cdu_f
from ..bases import CoreTrainer


class ClassificationTrainer(CoreTrainer):
    """CoreTrainer extended to classification tasks.

    Attributes
    ----------
    soft : torch.tensor
        Accumulates within-epoch soft (probability) confusion matrices.
    hard : torch.tensor
        Accumulates within epoch hard (discrete) confusion matrices.
    zarr (zarr.Array):
        Stores confusion matrices in `soft` and `hard` after each epoch. Has
        shape (n_folds, n_curve_steps, n_epochs, 2, 2, `n_class`, `n_class`),
        where...
        -   4th dimension corresponds to training (idx 0) and validation
            (idx 1) phases
        -   5th dimension corresponds to soft (idx 0) and
            hard (idx 1) confusion matrices.

    Parameters
    ----------
    n_class : int
        Number of classes predicted by model.
    **kwargs
        Passed to CoreTrainer.

    See Also
    --------
    cadlabutils.ml.bases.CoreTrainer : Parent class.
    """

    def __init__(
            self,
            n_class: int,
            **kwargs
    ):
        super(ClassificationTrainer, self).__init__(**kwargs)

        # model-specific template with which to generate confusion matrices
        self.soft = torch.zeros(
            (n_class, n_class), dtype=self.dtypes[0], device=self.device,
            requires_grad=False)
        self.hard = torch.zeros(
            (n_class, n_class), dtype=self.dtypes[0], device=self.device,
            requires_grad=False)
        shape = (1, 1, 1, 2, 2, n_class, n_class)
        self.zarr = cdu_f.zarrs.make_zarr(
            self.my_dir.joinpath("confusion_matrices.zarr"), shape=shape,
            chunk=shape, mode="a", dtype=float)

    def _step_stats(
            self,
            output,
            target
    ):
        """Compute summary statistics from current pass.

        Parameters
        ----------
        output
            Direct output of forward pass through `self.model`.
        target
            Ground truth target for comparison.

        Returns
        -------
        accuracy : float
            Accuracy of current prediction. Should be normalized to [0, 1]
            interval where 0 is worst possible performance and 1 is best.
        """
        accuracy, self.soft, self.hard = cdu.ml.classifier_stats(
            output, target, soft_mat=self.soft, hard_mat=self.hard)
        return accuracy

    def _epoch_reset(
            self,
            epoch: int,
            train: bool
    ):
        """Reset instance state after completing an epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        train : bool
            If True, training epoch. If False, validation epoch.
        """
        # prepare soft and hard confusion matrices
        stats = [m.detach().cpu().numpy() for m in (self.soft, self.hard)]
        norms = [np.linalg.norm(m, axis=1, keepdims=True) for m in stats]
        stats = [np.divide(m, n, where=n > 0) for m, n in zip(stats, norms)]
        stats = np.stack(stats, axis=0)

        # update zarr file to appropriate dimensionality
        new_shape = np.array(self.zarr.shape)
        new_shape[:3] = np.maximum(
            new_shape[:3], np.array(self.coords + [epoch]) + 1)
        self.zarr.resize(new_shape)

        # save confusion matrices and reset epoch holder values
        self.zarr[tuple(self.coords + [epoch, int(not train)])] = stats
        self.soft.zero_()
        self.hard.zero_()
