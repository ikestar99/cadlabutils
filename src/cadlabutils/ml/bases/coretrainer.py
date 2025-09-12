#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import numpy as np
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from pathlib import Path

import cadlabutils as cdu

from .. import utils


class CoreTrainer(ABC):
    """Helper class to facilitate inference with pytorch models.

    Class Attributes
    ----------------
    _MODE : tuple[str]
        Names of each mode of training.

    Attributes
    ----------
    model : torch.nn.Module
        Model to train and use for inference.
    criterion : torch.nn.loss._Loss
        Loss function with which to evaluate model performance.
    optimizer : torch.optim.Optimizer
        Optimizer with which to train the model.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        Scheduler with which to reduce learning rate over time.
    device : torch.device
        Hardware device used for inference.
    dtypes : tuple[torch.dtype, torch.dtype]
        Data typed of model and ground truth data.
    model_path : Path
        Path where model parameters are stored.
    config_json : Path
        Path where training hyperparameters are stored.

    Parameters
    ----------
    model : torch.nn.Module
        Model class.
    model_kwargs : dict
        Keyword arguments passed to `model` init.
    out_dir : pathlib.Path
        Directory in which to save training related files.
    criterion : type, optional
        Loss function class.
        Defaults to torch.nn.CrossEntropyLoss.
    optimizer : type, optional
        Optimizer class.
        Defaults to torch.optim.Adam.
    scheduler : type, optional
        Learning rate scheduler class.
        Defaults to nn.CrossEntropyLoss.
    gpu : int, optional
        CUDA device to use for inference.
        Defaults to 0.

    Other Parameters
    ----------------
    model_dtype : torch.dtype, optional
        Datatype of model.
        Defaults to torch.float32, or single-precision.
    target_dtype : torch.dtype, optional
        Datatype of ground truth. Should match loss function.
        Defaults to torch.int64.
    criterion_kwargs : dict, optional
        Keyword arguments passed to `criterion` init.
    optimizer_kwargs : dict, optional
        Keyword arguments passed to `optimizer` init.
        Defaults to {"lr": 1e-3"}.
    scheduler_kwargs : dict, optional
        Keyword arguments passed to `scheduler` init.
        Defaults to {"patience": 5, "threshold": 0.01}.

    See Also
    --------
    CoreTrainer._stats : abstract method, must be implemented by child class.
    CoreTrainer._reset : abstract method, must be implemented by child class.

    Notes
    -----
    `CoreTrainer` uses `torch.optim.lr_Scheduler.ReduceLROnPlateau` by default
    during training. To disable this feature, set `scheduler_kwargs` to
    {"patience": dummy_epochs} where `dummy_epochs` is an integer larger than
    the number of epochs planned during training.
    """

    def __init__(
            self,
            model: nn.Module,
            model_kwargs: dict,
            out_dir: Path,
            criterion: type = nn.CrossEntropyLoss,
            optimizer: type = torch.optim.Adam,
            scheduler: type = torch.optim.lr_scheduler.ReduceLROnPlateau,
            gpu: int = 0,
            model_dtype: torch.dtype = torch.float32,
            target_dtype: torch.dtype = torch.int64,
            criterion_kwargs: dict = None,
            optimizer_kwargs: dict = None,
            scheduler_kwargs: dict = None,
    ):
        # instantiate model and loss function
        self.model = model(*model_kwargs)
        self.criterion = criterion(**(criterion_kwargs or {}))

        # instantiate optimizer
        o_kwargs = {"params": self.model.parameters(), "lr": 1e-3}
        o_kwargs.update(optimizer_kwargs or {})
        self.optimizer = optimizer(**o_kwargs)

        # instantiate learning rate scheduler
        s_kwargs = {
            "optimizer": self.optimizer, "patience": 5, "threshold": 0.01}
        s_kwargs.update(scheduler_kwargs or {})
        self.scheduler = scheduler(**s_kwargs)

        # set auxiliary training variables
        self.device = utils.get_device(gpu)
        self.dtypes = (model_dtype, target_dtype)
        self.model_path = out_dir.joinpath("model.safetensors")
        self.config_json = out_dir.joinpath("config.json")

    @abstractmethod
    def _stats(
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
        float
            Accuracy of current prediction. Should be normalized to [0, 1]
            interval where 0 is worst possible performance and 1 is best.

        Notes
        -----
        Can also be used to generate per-epoch running statistics, such as a
        confusion matrix, with instance variables defined in `__init__`.
        """
        pass

    @abstractmethod
    def _reset(
            self,
            epoch: int,
            mode: str,
            loss: float,
            accuracy: float
    ):
        """Reset instance state after completing an epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        mode : str
            Type of current epoch. Native options are "train", "valid".
        loss : float
            Aggregated loss value for current epoch.
        accuracy : float
            Aggregated accuracy value for current epoch.

        Notes
        -----

        """
        pass

    def _step(
            self,
            sample: torch.tensor,
            target: torch.tensor,
            train: bool
    ):
        """Forward pass through model, prepare for backpropagation.

        Parameters
        ----------
        sample : torch.tensor
            Input data on which to run inference.
        target : torch.tensor, optional
            Corresponding ground truth labels.
        train : bool
            If True, prepare model for training. If False, prepare model for
            validation.

        Returns
        -------
        output : torch.tensor
            Output of forward pass through model.
        loss : torch.tensor
            Output of criterion.

        Notes
        -----
        Simple wrapper around `utils.forward_pass` utility function. For more
        complex training paradigms, subclasses should override this method with
        custom logic to account for different input and output structures. Any
        such function must:
        -   Accept 3 input arguments, `sample`, `target`, and `train`.
            The former two do not need to be tensors. The last, `train`, is a
            ``bool`` that set model to train mode for training and eval mode
            for validation.
        -   Return 2 output values. The first should be some output of forward
            pass through the model that can be compared to a ground truth for
            downstream accuracy measurement in `_statistics` method. The second
            must be a scalar loss tensor.

        Any such custom function must define both backpropagation
        (`loss.backward()') and optimization (`optim.step()`) steps.
        """
        output, loss = utils.forward_pass(
            self.model, sample, device=self.device, target=target,
            criterion=self.criterion,
            optimizer=self.optimizer if train else None,
            sample_dtype=self.dtypes[0], target_dtype=self.dtypes[1])
        return output, loss

    def _epoch(
            self,
            loader: torch.utils.data.DataLoader,
            train: bool,
            epoch: int
    ):
        """Complete a full iteration through dataset with model.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            Annotated dataset wrapped in a loader.
        train : bool
            If True, prepare model for training. If False, prepare model for
            validation.
        epoch : int
            Current epoch number.

        Returns
        -------
        running_stats : np.ndarray[float]
            Structure is (average_loss, average_accuracy) aggregated across all
            batches in `dataset`.
        """
        # prepare for training or inference
        self.model = utils.set_mode(
            self.model, train=train, device=self.device, dtype=self.dtypes[0])

        # loop over dataset once
        running_stats = []
        for sample, target in loader:
            # forward pass, backpropagation, optimization, and statistics
            output, loss = self._step(sample, target, train=train)
            running_stats += [[loss.item(), self._stats(output, target)]]

        # compute statistics and clean up after epoch
        running_stats = np.mean(np.array(running_stats), axis=0)
        self._reset(
            epoch, mode="train" if train else "valid", loss=running_stats[0],
            accuracy=running_stats[1])
        return running_stats

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        batch_size: int,
        epochs: int,
        resume: bool = False
    ):
        """
        Train a pytorch model on a preconfigured train/test dataset split via
        stochastic gradient descent.

        Args:
            train_dataset (torch.utils.data.Dataset):
                Annotated dataset on which to train model.
            valid_dataset (torch.utils.data.Dataset):
                Annotated dataset on which to validate model at the end of each
                epoch.
            batch_size (int):
                Number of samples to include in each forward pass. During
                training, model parameters are updates once per batch.
            epochs (int):
                Maximum number of training epochs during training. Each epoch
                involves a complete iteration through training and validation
                datasets.
            resume (bool, optional):
                If True, resume training from saved checkpoint.
                Defaults to False.
        """
        ep, op, sc = "epoch", "optimizer", "scheduler"
        epoch, t_acc, v_acc = 0, 0, 0

        # load model-specific checkpoint
        if resume and self.model_path.is_file():
            self.model, extras = utils.load(
                self.model_path, self.model, device=self.device,
                load_dict={op: self.optimizer, sc: self.scheduler})
            epoch += extras[E] + 1

        # prepare datasets
        train_loader = utils.get_dataloader(train_dataset, batch_size)
        valid_loader = utils.get_dataloader(valid_dataset, batch_size)

        # loop over full dataset per epoch
        for e in cdu.pbar(range(epoch, epochs), "epoch"):
            train_stats = self._epoch(train_loader, train=True, epoch=e)
            valid_stats = self._epoch(valid_loader, train=False, epoch=e)

            # modify learning rate based on validation loss
            self.scheduler.step(valid_stats[0])
            t_acc = max(train_stats[0], t_acc)
            v_acc = max(valid_stats[1], v_acc)
            if valid_stats[1] >= v_acc:
                utils.save(
                    self.model_path, self.model,
                    save_dict={op: self.optimizer, sc: self.scheduler}, ep=e)

        print(f"peak train: {t_acc:.2f} \npeak valid: {v_acc:.2f}")
