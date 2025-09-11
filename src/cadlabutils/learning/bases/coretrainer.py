#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    out_dir : pathlib.Path
        Directory in which to save model-related information.

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
        Defaults to torch.nn.CrossEntropyLoss.
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

    Notes
    -----
    `CoreTrainer` uses `torch.optim.lr_Scheduler.ReduceLROnPlateau` by default
    during training. To disable this feature, set `scheduler_kwargs` to
    {"patience": dummy_epochs} where `dummy_epochs` is an integer larger than
    the number of epochs planned during training.
    """
    _MODE = ("train", "valid")

    def __init__(
            self,
            model: nn.Module,
            model_kwargs: dict,
            out_dir: Path,
            criterion: type = nn.CrossEntropyLoss,
            optimizer: type = Adam,
            scheduler: type = ReduceLROnPlateau,
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
    def _statistics(
            self
    ):
        """compute statistics from current epoch."""
        pass

    def _step(
            self,
            sample: torch.tensor,
            target: torch.tensor
    ):
        """Forward pass through model, prepare for backpropagation.

        Parameters
        ----------
        sample : torch.tensor
            Input data on which to run inference.
        target : torch.tensor, optional
            Corresponding ground truth labels.

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
        custom logic to account for different input and output structures.
        Function should retain two outputs, the latter of which must be a
        scalar tensor with numerical loss value.

        Any such custom function must define both backpropagation
        (`loss.backward()') and optimization (`optim.step()`) steps.
        """
        output, loss = utils.forward_pass(
            self.model, sample, device=self.device, target=target,
            criterion=self.criterion, optimizer=self.optimizer,
            sample_dtype=self.dtypes[0], target_dtype=self.dtypes[1])
        return output, loss

    def test(
            self,
            dataset: Dataset,
            batch_size: int,
            epoch: int,
            train: bool
    ):
        """
        Test a trained model against an annotated test dataset.

        Args:
            dataset (Dataset):
                Annotated dataset on which to validate trained model after last
                epoch.
            batch_size (int):
                Number of samples to include in each forward pass. During
                training, model parameters are updates once per batch.
            epoch (int):
                Number of training epochs completed before test phase.

        Returns:
            (tuple):
            Contains the following two items:
            -    t_loss (float):
                    Average loss across all test samples.
            -   t_acc (float):
                    Average accuracy across all test samples.
        """
        # test phase
        self.model = utils.set_mode(
            self.model, train=train, device=self.device, dtype=self.dtypes[0])
        t_loss, t_corr, t_count, t_matrix = 0, 0, 0, self.template.clone()

        # loop over dataset once
        t_loss, t_acc = [], []
        loader = self._get_loader(dataset, batch_size)
        for sample, target in loader:
            # forward pass
            target, output, loss = self._step(sample, target)

            # collect running test statistics
            acc, t_matrix = self._step_stats(
                labels=target, output=output, matrix=t_matrix)
            t_loss, t_acc = t_loss + [loss.item()], t_acc + [acc]

        self._save_stats(
            epoch, mode=self._MODE[-1] if mode is None else mode, loss=t_loss,
            acc=t_acc, matrix=t_matrix)
        return np.mean(t_loss), np.mean(t_acc)

    def train(
        self,
        train_dataset: Dataset,
        valid_dataset: Dataset,
        batch_size: int,
        epochs: int,
        resume: bool = False
    ):
        """
        Train a pytorch model on a preconfigured train/test dataset split via
        stochastic gradient descent.

        Args:
            train_dataset (Dataset):
                Annotated dataset on which to train model.
            valid_dataset (Dataset):
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
        E, O, S = "epoch", "optimizer", "scheduler"
        # load model-specific checkpoint if it exists and resume training
        epoch = 0
        if resume:
            self.model, extras = utils.load(
                self.model_path, self.model, device=self.device,
                load_dict={O: self.optimizer, S: self.scheduler})

            epoch
        check = resume and self.save_safetensors.is_file()
        e = self._load_checkpoint() + 1 if check else 0
        my_peak = 0
        train_loader = self._get_loader(train_dataset, batch_size)
        for e in aau.pbar(range(e, epochs), "epoch"):
            t_loss, t_acc, t_matrix = [], [], self.template.clone()

            # training phase
            self._learning_mode()
            for sample, labels in train_loader:
                # forward pass
                output, loss = self._step(sample, labels)

                # collect running statistics
                acc, t_matrix = self._step_stats(
                    labels=labels, output=output, matrix=t_matrix)
                t_loss, t_acc = t_loss + [loss.item()], t_acc + [acc]

            # save train statistics
            self._save_stats(
                e, mode=self._MODE[0], loss=t_loss, acc=t_acc, matrix=t_matrix)

            # validation phase
            v_loss, v_acc = self.test(
                dataset=valid_dataset, batch_size=batch_size, epoch=e,
                mode=self._MODE[1])

            # modify learning rate based on validation loss, clean up
            self.sched.step(v_loss)
            if v_acc <= my_peak:
                continue

            my_peak = max(v_acc, my_peak)
            utils.save(
                self.model_path, self.model,
                save_dict={O: self.optimizer, S: self.scheduler}, E=e)

        print(f"peak validation/peak total: {my_peak:.2f} / {self.peak:.2f}")
