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
import cadlabutils.files as cdu_f

from .. import utils, metrics


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
    fold : int
        Current k-fold fold.
    curve : int
        Current index along learning curve generation.
    _cfg : dict
        Keyword arguments used to instantiate trainable parameters.

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
    dtypes : tuple[torch.dtype], optional
        Datatype of model and ground truth, matched to loss function.
        Defaults to (torch.float32, torch.int64) for CrossEntropyLoss.
    criterion_kwargs : dict, optional
        Keyword arguments passed to `criterion` init.
    optimizer_kwargs : dict, optional
        Keyword arguments passed to `optimizer` init.
        Defaults to {"lr": 1e-3"}.
    scheduler_kwargs : dict, optional
        Keyword arguments passed to `scheduler` init.
        Defaults to {"patience": 5}.

    See Also
    --------
    CoreTrainer._step_stats : abstract method, must be implemented by child.
    CoreTrainer._epoch_reset : abstract method, must be implemented by child.
    CoreTrainer._make_plots: abstract method, must be implemented by child.

    Notes
    -----
    `CoreTrainer` uses `torch.optim.lr_Scheduler.ReduceLROnPlateau` by default
    during training. To disable this feature, set `scheduler_kwargs` to
    {"patience": dummy_epochs} where `dummy_epochs` is an integer larger than
    the number of epochs planned during training.
    """
    _BAR, _CPU, _GPU, _GPR = None, None, None, None

    def __init__(
            self,
            model: nn.Module,
            model_kwargs: dict,
            out_dir: Path,
            criterion: type = nn.CrossEntropyLoss,
            optimizer: type = torch.optim.Adam,
            scheduler: type = torch.optim.lr_scheduler.ReduceLROnPlateau,
            gpu: int = 0,
            dtypes: tuple[torch.dtype] = (torch.float32, torch.int64),
            criterion_kwargs: dict = None,
            optimizer_kwargs: dict = None,
            scheduler_kwargs: dict = None,
    ):
        # set instance variables
        self.device = utils.get_device(gpu)
        self.dtypes = dtypes
        self.model_path = out_dir.joinpath("model.safetensors")
        self.fold, self.curve, self.batch_size = 0, 0, None

        # prepare reset dict and initialize trainable parameters
        self._cfg = {
            "model": (model, model_kwargs),
            "criterion": (criterion, criterion_kwargs or {}),
            "optimizer": (optimizer, {"lr": 1e-3, **(optimizer_kwargs or {})}),
            "scheduler": (
                scheduler, {"patience": 5, **(scheduler_kwargs or {})})}

        config_yaml = out_dir.joinpath("config.yaml")
        if config_yaml.is_file():  # load existing configuration
            config = cdu_f.yamls.from_yaml(config_yaml)
            for k, (_, v) in config.items():
                self._cfg[k] = (self._cfg[k][0], v)
        else:  # save configuration for reuse
            out_dir.mkdir(exist_ok=True, parents=True)
            cdu_f.yamls.to_yaml(
                config_yaml, {
                    k: [f"{c.__module__}.{c.__qualname__}", d]
                    for k, (c, d) in self._cfg.items()})

        self._initialize()

    @abstractmethod
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
        float
            Accuracy of current prediction. Should be normalized to [0, 1]
            interval where 0 is worst possible performance and 1 is best.

        Notes
        -----
        Must be implemented by child classes.
        Can also be used to generate per-epoch running statistics, such as a
        confusion matrix, with instance variables defined in `__init__`.
        """
        pass

    @abstractmethod
    def _epoch_reset(
            self,
            epoch: int,
            train: bool,
            samples: int,
            loss: float,
            accuracy: float
    ):
        """Reset instance state after completing an epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        train : bool
            If True, training epoch. If False, validation epoch.
        samples : int
            Number of samples in current dataset.
        loss : float
            Mean loss value across current dataset.
        accuracy : tuple[float, float]
            Mean accuracy value across current dataset..

        Notes
        -----
        Must be implemented by child classes.
        """
        pass

    @abstractmethod
    def _make_plots(
            self
    ):
        """Generate graphs at the end of training.

        Notes
        -----
        Must be implemented by child classes.
        Generate plot of statistics saved during training.
        """
        pass

    def _track_memory(
            self
    ):
        """Track resource consumption on CPU and CUDA device if available.

        See Also
        --------
        cadlabutils.TreeBar

        Notes
        -----
        Depends on class attribute `_BAR`, which is a `Treebar` instance set
        during the training loop.
        """
        if self._BAR is None:
            return

        c_u, c_t = cdu.get_cpu_memory(scale=3)
        self._CPU = (
            self._BAR.add_task("CPU use (GiB)", tabs=1, total=c_t)
            if self._CPU is None else self._CPU)
        self._BAR.update(self._CPU, completed=c_u)
        if self. device.type == "cpu":
            return

        g_u, g_r, g_t = utils.get_cuda_memory(self.device, scale=3)
        if self._GPU is None:
            self._GPU = self._BAR.add_task("GPU use (GiB)", tabs=2, total=g_t)
            self._GPR = self._BAR.add_task("GPU res (GiB)", tabs=2, total=g_t)

        self._BAR.update(self._CPU, completed=c_u)
        self._BAR.update(self._GPR, completed=g_r)

    def _initialize(
            self
    ):
        """Reinitialize trainable parameters."""
        self.model = utils.set_mode(
            self._cfg["model"][0](**self._cfg["model"][1]),
            train=True, device=self.device, dtype=self.dtypes[0])
        self.criterion = self._cfg["criterion"][0](**self._cfg["criterion"][1])
        self.optimizer = self._cfg["optimizer"][0](
            params=self.model.parameters(), **self._cfg["optimizer"][1])
        self.scheduler = self._cfg["scheduler"][0](
            optimizer=self.optimizer, **self._cfg["scheduler"][1])

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
        target : torch.tensor
            Corresponding ground truth labels moved to inference device.

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
        output, loss, target = utils.forward_pass(
            self.model, sample, device=self.device, target=target,
            criterion=self.criterion,
            optimizer=self.optimizer if train else None,
            sample_dtype=self.dtypes[0], target_dtype=self.dtypes[1])
        return output, loss, target

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
        agg_loss : float
            Loss, averaged across all batches in `dataset`.
        agg_loss : float
            Accuracy, averaged across all batches in `dataset`.
        """
        torch.cuda.empty_cache()

        # prepare for training or inference
        self.model = utils.set_mode(
            self.model, train=train, device=self.device, dtype=self.dtypes[0])

        # loop over dataset once
        running_stats = []
        for sample, target in loader:
            # forward pass, backpropagation, optimization, and statistics
            output, loss, target = self._step(sample, target, train=train)
            running_stats += [[loss.item(), self._step_stats(output, target)]]
            self._track_memory()
            del output, loss, target

        # compute statistics and clean up after epoch
        agg_loss, agg_acc = np.mean(np.array(running_stats), axis=0)
        self._epoch_reset(
            epoch, train=train, samples=len(loader.dataset), loss=agg_loss,
            accuracy=agg_acc)
        return agg_loss, agg_acc

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        epochs: int,
        fold: int = 0,
        curve: int = 0,
        pbar: cdu.TreeBar = None,
        epoch_task: cdu.rp.TaskID = None
    ):
        """Train a pytorch model on a preconfigured train/test dataset split.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            Annotated dataset on which to train model.
        valid_dataset : torch.utils.data.Dataset
            Annotated dataset on which to validate model at the end of each
            epoch.
        epochs : int
            Maximum number of training epochs during training. Each epoch
            involves a complete iteration through training and validation
            datasets.
        fold : int, optional
            Current k-fold fold.
            Defaults to 0.
        curve : int, optional
            Current index along learning curve generation.
            Defaults to 0.

        Returns
        -------
        t_max : float
            Peak average training accuracy observed across epochs.
        v_max : float
            Peak average validation accuracy observed across epochs.

        Other Parameters
        ----------------
        pbar : cdu.TreeBar, optional
            Progress bar displaying current epoch information.
            Defaults to None, in which case no epoch progress bar is displayed.
        epoch_task : cdu.rp.TaskID, optional
            Index of epoch progress bar in `tree_bar`.
            Defaults to None.
        """
        self.fold, self.curve, self._BAR = fold, curve, pbar
        op, sc = "optimizer", "scheduler"
        epoch, t_max, v_max = 0, 0, 0

        # simulate optimum batch size
        self._initialize()
        self._track_memory()
        if self.batch_size is None:
            pair = train_dataset[0]
            self.batch_size = metrics.simulate_batch_size(
                self.model, sample=pair[0], device=self.device, target=pair[1],
                criterion=self.criterion, optimizer=self.optimizer,
                sample_dtype=self.dtypes[0], target_dtype=self.dtypes[1])
            print("Simulated batch size:", self.batch_size)

        # load model-specific checkpoint
        if self.model_path.is_file():
            extras = utils.load(
                self.model_path, self.model, device=self.device,
                load_dict={op: self.optimizer, sc: self.scheduler})
            epoch = extras["epoch"] + 1

            # skip if current fold/curve already completed
            if extras["fold"] < fold or extras["curve"] < curve:
                self._make_plots()
                return None, None

        # prepare datasets
        train_loader = utils.get_dataloader(train_dataset, self.batch_size)
        valid_loader = utils.get_dataloader(valid_dataset, self.batch_size)

        # loop over full dataset per epoch
        for e in range(epoch, epochs):
            t_loss, t_acc = self._epoch(train_loader, train=True, epoch=e)
            v_loss, v_acc = self._epoch(valid_loader, train=False, epoch=e)

            # modify learning rate based on validation loss
            self.scheduler.step(v_loss)
            t_max = max(t_acc, t_max)
            v_max = max(v_acc, v_max)
            if self._BAR is not None:
                label = f"{len(train_loader)} train batches"
                delta = epoch if e == epoch and e != 0 else 0
                pbar.update(epoch_task, advance=1 + delta, label=label)
            if v_acc >= v_max:
                utils.save(
                    self.model_path, self.model,
                    save_dict={op: self.optimizer, sc: self.scheduler},
                    epoch=e, fold=fold, curve=curve)

        self._make_plots()
        del self.model, self.criterion, self.optimizer, self.scheduler
        torch.cuda.empty_cache()
        return t_max, v_max
