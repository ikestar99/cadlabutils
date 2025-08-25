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

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import arealize.utils as aau
import arealize.models as aam


class Trainer:
    """
    Helper class to facilitate inference with pytorch models.

    Class Attributes:
        _MODE (tuple[str]):
            Names of each mode of training.
        _COLS (list[int]):
            Columns names of training statistics file (csv).

    Attributes:
        device (torch.device):
            Hardware device used for inference.
        fold (int):
            Index of current model in k-fold split.
        model (torch.nn.Module):
            Model to train and use for inference.
        critr (torch.nn.loss._Loss):
            Loss function with which to evaluate model performance.
        optim (torch.optim.Optimizer):
            Optimizer with which to train the model.
        sched (torch.optim.lr_scheduler._LRScheduler):
            Scheduler with which to reduce learning rate over time.
        checkpoint_pth (pathlib.Path):
            Path to save model checkpoints (pth).
        statistics_csv (pathlib.Path):
            Path to save running statistics during training (csv).
        template (torch.tensor):
            Confusion matrix template. Used to track running statistics during
            each mode within an epoch.
        peak (float):
            Peak validation accuracy seen thusfar.
    """
    _MODE = ("train", "valid", "test")
    _CN = "model_name"
    _CF = "fold"
    _CS = "samples"
    _CE = "epoch"
    _CM = "mode"
    _CL = "loss"
    _CLS = "loss_sem"
    _CA = "acc"
    _CAS = "acc_sem"
    _CGT = "labels"
    _ALL = [_CN, _CF, _CS, _CE, _CM, _CL, _CLS, _CA, _CAS, _CGT]

    def __init__(
            self,
            model: nn.Module,
            name: str,
            fold: int,
            samples: int,
            classes: int,
            gpu: int = 0,
            lr: float = 1e-3,
            criterion: type = nn.CrossEntropyLoss,
            optimizer: type = torch.optim.Adam,
            cr_kwargs: dict = None,
            op_kwargs: dict = None,
            schedule: bool = True,
    ):
        """
        Instantiate Predictor with known model.

        Args:
            model (torch.nn.Module):
                Instantiated model to use for inference.
            name (str):
                Used to identify model in output files.
            fold (int):
                Index of current model in k-fold split.
            samples (int):
                Number of samples used for training.
            classes (int):
                Number of output classes predicted by model.
            gpu (int, optional):
                CUDA device to use for inference.
                Defaults to 0.
            lr (float, optional):
                Initial learning rate for optimizer.
                Defaults to 1e-3.
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
        """
        # identify hardware device on which to run inference
        self._get_device(gpu)

        # set instance attributes
        self.fold = fold
        self.samples = samples

        # instantiate training components with associated arguments
        self.model = model.double().to(self.device)
        self.critr = criterion(**({} if cr_kwargs is None else cr_kwargs))
        self.optim = optimizer(
            params=self.model.parameters(), lr=lr,
            **({} if op_kwargs is None else op_kwargs))
        self.sched = None if not schedule else ReduceLROnPlateau(
            optimizer=self.optim, mode="min", factor=0.1, patience=5,
            threshold=0.01)

        # generate output data paths
        self.checkpoint_pth = aam.MODEL_DIR.joinpath(f"{name}.pth")
        self.statistics_csv = aam.STATISTIC_DIR.joinpath(
            f"{name}_statistics.csv")

        # model-specific template with which to generate confusion matrices
        self.template = torch.zeros(
            (classes, classes), dtype=torch.double, device=self.device,
            requires_grad=False)

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

    @classmethod
    def plot_stats(
            cls,
            file_csv: str,
            epochs: int,
            x_axis: str
    ):
        """
        Plot loss and accuracy metrics across training epochs.

        Args:
            file_csv (str):
                Path to CSV file containing training statistics.
            epochs (int):
                Max number of training epochs.
            x_axis (str):
                Variable to plot on x-axis. Options are "epoch" (self._CE) or
                "samples" (self._CS).
        """
        # load statistics
        stats = pd.read_csv(file_csv, usecols=cls._ALL[:-1]).drop_duplicates()

        # normalize loss to maximum value, pivot to long form
        # stats[[cls._CL, cls._CLS]] /= stats[cls._CL].max()
        stats = stats.melt(
            id_vars=[cls._CN, cls._CS, cls._CE, cls._CM], var_name="stat",
            value_vars=[cls._CL, cls._CA], value_name="normalized value")

        # set style, colors, and create line plot
        sns.set_theme(style="ticks", rc={
            "axes.spines.right": False, "axes.spines.top": False})
        g = sns.relplot(
            data=stats, x=x_axis, y="normalized value", row="stat",
            col=cls._CN, hue=cls._CM, errorbar=("se", 2), kind="line",
            aspect=2, palette=sns.color_palette("rocket", n_colors=3))

        # clean up and plot
        x_min = 0 if x_axis == cls._CE else 1
        x_max = epochs - 1 if x_axis == cls._CE else stats[cls._CS].max()
        g.set(xlim=(x_min, x_max), ylim=(0, 1))
        plt.show()

    @staticmethod
    def _get_loader(
            dataset: Dataset,
            batch_size: int,
            workers: int = 4
    ):
        """
        Create a DataLoader from a Dataset.

        Args:
            dataset (Dataset):
                Dataset to wrap in a DataLoader.
            batch_size (int):
                Number of samples per batch.
            workers (int, optional):
                Number of parallel workers.
                Defaults to 4.

        Returns:
            loader (DataLoader):
                Instantiated DataLoader.
        """
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
            pin_memory=True, persistent_workers=True)
        return loader

    @staticmethod
    def _step_stats(
            labels: torch.tensor,
            output: torch.tensor,
            matrix: torch.tensor
    ):
        """
        Compute statistics after forward pass through model.

        Args:
            labels (torch.tensor):
                Ground truth labels of current sample. Has shape (batch, ...).
            output (torch.tensor):
                Output of model(sample). Has shape (batch, classes, ...) where
                trailing dimensions match those of corresponding labels.
            matrix (torch.tensor):
                Running confusion matrix of class-wise prediction
                probabilities. Has shape (true classes, predicted classes).

        Returns:
            (tuple):
                Contains the following two items:
                -   (float):
                        Prediction accuracy averaged across batches.
                -   matrix (torch.tensor):
                        Updated confusion matrix of class-wise prediction
                        probabilities.
        """
        # convert logits into probabilities and calculate correct count
        output = F.softmax(output, dim=1)
        correct = int((torch.argmax(output, dim=1) == labels).sum().item())

        # update confusion matrix probability distributions per class
        labels = labels.reshape(-1)
        output = torch.movedim(output, 1, -1).reshape(-1, output.size(1))
        for c in range(matrix.size(0)):
            matrix[c] += output[labels == c].sum(dim=0)

        return correct / labels.numel(), matrix

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
                shape (true classes, predicted classes).
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

    def _get_device(
            self,
            gpu: int
    ):
        """
        Identify a CUDA-enabled device on which to perform parallelized
        computations.

        Args:
            gpu (int):
                Index of CUDA device to use. Will default to highest available
                index if provided integer is out of bounds.
        """
        self.device = torch.device(
            f"cuda:{min(gpu, torch.cuda.device_count() - 1)}"
            if torch.cuda.is_available() and gpu is not None else "cpu")

    def _save_checkpoint(
            self,
            epoch: int,
    ):
        """
        Save model parameters in a checkpoint.pth file.

        Args:
            epoch (int):
                Training epoch after which to save model parameters.
        """
        saved = {} if not self.checkpoint_pth.is_file() else torch.load(
            self.checkpoint_pth)
        states = {
            "model": self.model.state_dict(), "epoch": epoch,
            "optimizer": (
                self.optim.state_dict() if self.optim is not None else None),
            "scheduler": (
                self.sched.state_dict() if self.sched is not None
                else None)}
        saved[self.model_name] = {
            k: v for k, v in states.items() if v is not None}
        torch.save(saved, self.checkpoint_pth)

    def _load_checkpoint(
            self
    ):
        """
        Load model parameters from a checkpoint.pth file.
        """
        states = torch.load(
            self.checkpoint_pth, map_location=self.device)[self.model_name]
        self.model.load_state_dict(states["model"])
        if "optimizer" in states and self.optim is not None:
            self.optim.load_state_dict(states["optimizer"])
        if "scheduler" in states and self.sched is not None:
            self.sched.load_state_dict(states["scheduler"])

        return states["epoch"]

    def _learning_mode(
            self
    ):
        """
        Prepare for backpropagation by enabling gradients.
        """
        self.model = self.model.train()
        torch.set_grad_enabled(True)

    def _evaluate_mode(
            self
    ):
        """
        Prepare for inference by disabling gradients and computation graph.
        """
        self.model = self.model.eval()
        torch.set_grad_enabled(False)

    def _step(
            self,
            sample: torch.tensor,
            labels: torch.tensor = None
    ):
        """
        Run inference and compute loss if labels are available.

        Args:
            sample (torch.tensor):
                Input data on which to run inference.
            labels (torch.tensor, optional):
                Ground truth labels corresponding to the input samples.
                Defaults to None, in which case loss is not computed.

        Returns:
            (tuple):
                Contains the following 3 items:
                -   0 (torch.tensor):
                        Ground truth labels moved to appropriate device.
                -   1 (torch.tensor):
                        Model prediction after forward pass.
                -   2 (torch.tensor):
                        Loss.
        """
        # forward pass, predict.shape = (batch_size, n_classes)
        labels = labels.long().to(self.device, non_blocking=True)
        output = self.model(sample.double().to(self.device, non_blocking=True))
        loss = self.critr(output, labels)
        return labels, output, loss

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
        check = resume and self.checkpoint_pth.is_file()
        e = self._load_checkpoint() + 1 if check else 0
        my_peak = 0
        train_loader = self._get_loader(train_dataset, batch_size)
        for e in aau.pbar(range(e, epochs), "epoch"):
            t_loss, t_acc, t_matrix = [], [], self.template.clone()

            # training phase
            self._learning_mode()
            for sample, labels in train_loader:
                # forward pass
                self.optim.zero_grad()
                labels, output, loss = self._step(sample, labels)

                # back propagation
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optim.step()

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
            if v_acc > my_peak:
                my_peak = max(v_acc, my_peak)
                self._save_checkpoint(e)

        print(f"peak validation/peak total: {my_peak:.2f} / {self.peak:.2f}")
