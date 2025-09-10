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
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utils import *


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
    critr : torch.nn.loss._Loss
        Loss function with which to evaluate model performance.
    optim : torch.optim.Optimizer
        Optimizer with which to train the model.
    sched : torch.optim.lr_scheduler._LRScheduler
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
        self.critr = criterion(**(criterion_kwargs or {}))

        # instantiate optimizer
        o_kwargs = {"params": self.model.parameters(), "lr": 1e-3}
        o_kwargs.update(optimizer_kwargs or {})
        self.optim = optimizer(**o_kwargs)

        # instantiate learning rate scheduler
        s_kwargs = {"optimizer": self.optim, "patience": 5, "threshold": 0.01}
        s_kwargs.update(scheduler_kwargs or {})
        self.sched = scheduler(**s_kwargs)

        # set auxiliary training variables
        self.device = get_device(gpu)
        self.dtypes = (model_dtype, target_dtype)
        self.out_dir = out_dir

    @abstractmethod
    def _save_stats(
            self,
            *args,
            **kwargs
    ):
        """compute statistics from current epoch."""
        pass

    def _step(
            self,
            sample: torch.tensor,
            target: torch.tensor
    ):
        output, target, loss = forward_pass(
            self.model, sample, device=self.device, target=target,
            criterion=self.critr, optimizer=self.optim,
            sample_dtype=self.dtypes[0], target_dtype=self.dtypes[1])
        return output, target, loss


    def _load_checkpoint(
            self
    ):
        """
        Load model parameters from a checkpoint.pth file.
        """
        states = torch.load(
            self.save_safetensors, map_location=self.device)[self.model_name]
        self.model.load_state_dict(states["model"])
        if "optimizer" in states and self.optim is not None:
            self.optim.load_state_dict(states["optimizer"])
        if "scheduler" in states and self.sched is not None:
            self.sched.load_state_dict(states["scheduler"])

        return states["epoch"]

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
            mode (str):
                Mode of test phase.
                Defaults to None, in which case mode is Predictor._MODE[-1].

        Returns:
            (tuple):
            Contains the following two items:
            -    t_loss (float):
                    Average loss across all test samples.
            -   t_acc (float):
                    Average accuracy across all test samples.
        """
        # test phase
        self.model = set_mode(
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
        # load model-specific checkpoint if it exists and resume training
        check = resume and self.save_safetensors.is_file()
        e = self._load_checkpoint() + 1 if check else 0
        my_peak = 0
        train_loader = self._get_loader(train_dataset, batch_size)
        # for e in aau.pbar(range(e, epochs), "epoch"):
        #     t_loss, t_acc, t_matrix = [], [], self.template.clone()
        #
        #     # training phase
        #     self._learning_mode()
        #     for sample, labels in train_loader:
        #         # forward pass
        #         self.optim.zero_grad()
        #         labels, output, loss = self._step(sample, labels)
        #
        #         # back propagation
        #         loss.backward()
        #         nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        #         self.optim.step()
        #
        #         # collect running statistics
        #         acc, t_matrix = self._step_stats(
        #             labels=labels, output=output, matrix=t_matrix)
        #         t_loss, t_acc = t_loss + [loss.item()], t_acc + [acc]
        #
        #     # save train statistics
        #     self._save_stats(
        #         e, mode=self._MODE[0], loss=t_loss, acc=t_acc, matrix=t_matrix)
        #
        #     # validation phase
        #     v_loss, v_acc = self.test(
        #         dataset=valid_dataset, batch_size=batch_size, epoch=e,
        #         mode=self._MODE[1])
        #
        #     # modify learning rate based on validation loss, clean up
        #     self.sched.step(v_loss)
        #     if v_acc > my_peak:
        #         my_peak = max(v_acc, my_peak)
        #         self._save_checkpoint(e)
        #
        # print(f"peak validation/peak total: {my_peak:.2f} / {self.peak:.2f}")
