#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


# 2. Third-party library imports
import torch

# 3. Local application / relative imports
from ..bases import CoreTrainer


class SegmentationTrainer(CoreTrainer):
    """CoreTrainer extended to Semantic Segmentation tasks.

    Attributes
    ----------

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
            *args,
            dtypes: tuple[torch.dtype] = (torch.float32, torch.float32),
            **kwargs
    ):
        super(SegmentationTrainer, self).__init__(
            *args, dtypes=dtypes, **kwargs)

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
        output = torch.argmax(output, dim=1)
        target = target if target.ndim == output.ndim else torch.argmax(
            target, dim=1)
        accuracy = torch.mean((output == target).float()).item()
        return accuracy

    def _epoch_reset(
            self,
            epoch: int,
            subset: int,
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
        pass
