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

from torch.optim import Adam
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


from ..utils import *


class CoreTrainer:
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
    save_safetensors : pathlib.Path
        Path to save model checkpoints (safetensors).
    """
    _MODE = ("train", "valid")

    def __init__(
            self,
            model: nn.Module,
            name: str,
            criterion: type = nn.CrossEntropyLoss,
            optimizer: type = Adam,
            scheduler: type = ReduceLROnPlateau,
            criterion_kwargs: dict = None,
            optimizer_kwargs: dict = None,
            scheduler_kwargs: dict = None,
            gpu: int = 0,
            lr: float = 1e-3,
            schedule: bool = True,
    ):
        """
        Instantiate Predictor with known model.

        Args:
            model (torch.nn.Module):
                Instantiated model to use for inference.
            name (str):
                Used to identify model in output files.
            criterion (type, optional):
                Loss function to use for inference.
                Defaults to torch.nn.CrossEntropyLoss.
            optimizer (type, optional):
                Optimizer to use for inference.
                Defaults to torch.optim.Adam.
            cr_kwargs (dict, optional):
                Keyword arguments to pass to criterion.__init__ function call.
                Defaults to None, in which case no keyword arguments are
                passed.
            op_kwargs (dict, optional):
                Keyword arguments to pass to optimizer.__init__ function call.
                Defaults to None, in which case no keyword arguments are
                passed.
            schedule (bool, optional):
                If True, use a scheduler to lower learning rate when loss
                plateaus.
                Defaults to True.
            gpu (int, optional):
                CUDA device to use for inference.
                Defaults to 0.
            lr (float, optional):
                Initial learning rate for optimizer.
                Defaults to 1e-3.
        """
        # identify hardware device on which to run inference
        self.device = get_device(gpu)

        # instantiate training components with associated arguments
        self.model = model.double().to(self.device)
        self.critr = criterion(**(criterion_kwargs or {}))

        o_kwargs = {"params": self.model.parameters(), "lr": 1e-3}
        o_kwargs.update(optimizer_kwargs or {})
        self.optim = optimizer(**o_kwargs)

        s_kwargs = {
            "optimizer": self.optim, "mode": "min", "factor": 0.1,
            "patience": 5, "threshold": 0.01}
        s_kwargs.update(scheduler_kwargs or {})
        self.sched = ReduceLROnPlateau(**s_kwargs)

        # # generate output data paths
        self.save_safetensors = aam.MODEL_DIR.joinpath(f"{name}.pth")
        self.statistics_csv = aam.STATISTIC_DIR.joinpath(
            f"{name}_statistics.csv")

    def __getattr__(
            self,
            name: str
    ):
        if name == "model_name":
            return self.model.__class__.__name__
        elif name == "statistics":
            attribute = pd.read_csv(self.statistics_csv)
        elif name == "peak":
            check = self.statistics_csv.is_file()
            attribute = self.statistics[self._CA].max() if check else 0
        else:
            raise AttributeError(f"Trainer has no attribute named {name}")

        return attribute

    # @classmethod
    # def plot_stats(
    #         cls,
    #         file_csv: str,
    #         epochs: int,
    #         x_axis: str
    # ):
    #     """
    #     Plot loss and accuracy metrics across training epochs.
    #
    #     Args:
    #         file_csv (str):
    #             Path to CSV file containing training statistics.
    #         epochs (int):
    #             Max number of training epochs.
    #         x_axis (str):
    #             Variable to plot on x-axis. Options are "epoch" (self._CE) or
    #             "samples" (self._CS).
    #     """
    #     # load statistics
    #     stats = pd.read_csv(file_csv, usecols=cls._ALL[:-1]).drop_duplicates()
    #
    #     # normalize loss to maximum value, pivot to long form
    #     # stats[[cls._CL, cls._CLS]] /= stats[cls._CL].max()
    #     stats = stats.melt(
    #         id_vars=[cls._CN, cls._CS, cls._CE, cls._CM], var_name="stat",
    #         value_vars=[cls._CL, cls._CA], value_name="normalized value")
    #
    #     # set style, colors, and create line plot
    #     sns.set_theme(style="ticks", rc={
    #         "axes.spines.right": False, "axes.spines.top": False})
    #     g = sns.relplot(
    #         data=stats, x=x_axis, y="normalized value", row="stat",
    #         col=cls._CN, hue=cls._CM, errorbar=("se", 2), kind="line",
    #         aspect=2, palette=sns.color_palette("rocket", n_colors=3))
    #
    #     # clean up and plot
    #     x_min = 0 if x_axis == cls._CE else 1
    #     x_max = epochs - 1 if x_axis == cls._CE else stats[cls._CS].max()
    #     g.set(xlim=(x_min, x_max), ylim=(0, 1))
    #     plt.show()

    def _save_stats(
            self,
            epoch: int,
            mode: str,
            loss: list,
            acc: list,
            matrix: torch.tensor,
    ):
        """
        Save statistics from training, validation, or test phases of model
        inference.

        Args:
            epoch (int):
                Current epoch of training.
            mode (str):
                Current phase of training. Value in Predictor._MODE attribute.
            loss (list):
                Loss per batch.
            acc (list):
                Accuracy per batch.
            matrix (torch.tensor):
                Confusion matrix of class-wise prediction probabilities. Has
                shape (true bases, predicted bases).
        """
        # convert confusion matrix to DataFrame on CPU
        matrix = matrix.detach().cpu().numpy()
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        cols = [f"class_{c}" for c in range(matrix.shape[0])]
        matrix = pd.DataFrame(matrix, columns=cols)

        # add running statistics and metadata to DataFrame
        matrix = matrix.assign(**{
            self._CN: self.model_name, self._CF: self.fold,
            self._CS: self.samples, self._CE: epoch, self._CM: mode,
            self._CL: np.mean(loss), self._CLS: 2 * ss.sem(loss),
            self._CA: np.mean(acc), self._CAS: 2 * ss.sem(acc),
            self._CGT: np.arange(matrix.shape[0])})

        # append statistics to existing csv file, if it exists
        check = self.statistics_csv.is_file()
        matrix[self._ALL + cols].to_csv(
            self.statistics_csv, header=not check, index=False,
            mode="a" if check else "w")

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
            mode: str = None
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
        self._evaluate_mode()
        t_loss, t_corr, t_count, t_matrix = 0, 0, 0, self.template.clone()

        # loop over dataset once
        t_loss, t_acc = [], []
        loader = self._get_loader(dataset, batch_size)
        for sample, labels in loader:
            # forward pass
            labels, output, loss = self._step(sample, labels)

            # collect running test statistics
            acc, t_matrix = self._step_stats(
                labels=labels, output=output, matrix=t_matrix)
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
