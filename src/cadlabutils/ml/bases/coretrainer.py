#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


# 1. Standard library imports
from abc import ABC, abstractmethod
from pathlib import Path

# 2. Third-party library imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 3. Local application / relative imports
import cadlabutils as cdu
import cadlabutils.files as cdu_f
from .. import metrics, utils


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
    curr_path : Path
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
    name : str
        Unique identifier for current model.
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
    pbar : cdu.TreeBar, optional
        Progress bar displaying current epoch information.
        Defaults to None, in which case no epoch progress bar is displayed.

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
    _RAM, _GPU, _GPR = None, None, None
    _M, _C, _O, _S = "model", "criterion", "optimizer", "scheduler"
    COLS = [
        "model", "name", "fold", "curve", "n", "epoch", "mode", "loss", "acc"]

    def __init__(
            self,
            model: nn.Module,
            model_kwargs: dict,
            out_dir: Path,
            name: str,
            criterion: type = nn.CrossEntropyLoss,
            optimizer: type = torch.optim.Adam,
            scheduler: type = torch.optim.lr_scheduler.ReduceLROnPlateau,
            gpu: int = 0,
            dtypes: tuple[torch.dtype] = (torch.float32, torch.int64),
            criterion_kwargs: dict = None,
            optimizer_kwargs: dict = None,
            scheduler_kwargs: dict = None,
            pbar: cdu.TreeBar = None,
    ):
        name = cdu.clean_name(Path(name)).stem
        out_dir = out_dir.joinpath("models")
        my_dir = out_dir.joinpath(name)

        # set instance variables
        self.device = utils.get_device(gpu)
        self.dtypes = dtypes
        self.curr_path = my_dir.joinpath(f"{name}_ckpt.safetensors")
        self.peak_path = my_dir.joinpath(f"{name}_peak.safetensors")
        self.stat_csv = out_dir.joinpath("coretrainer_stats.csv")
        self.batch_size = None
        self.pbar = pbar
        self.v_min, self.v_max = None, 0
        self.names, self.coords = [model.__name__, name], [0, 0]

        # load data from prior model runs
        self.stat = None
        if self.stat_csv.is_file():
            stat = pd.read_csv(self.stat_csv)
            self.stat = stat.loc[
                (stat[self.COLS[0]] == self.names[0]) &
                (stat[self.COLS[1]] == self.names[1])]

        # prepare reset dict and initialize trainable parameters
        self._cfg = {
            self._M: (model, model_kwargs),
            self._C: (criterion, criterion_kwargs or {}),
            self._O: (optimizer, {"lr": 1e-3, **(optimizer_kwargs or {})}),
            self._S: (scheduler, {"patience": 1, **(scheduler_kwargs or {})})}

        # load existing configuration
        config_yaml = out_dir.joinpath(f"{name}_config.yaml")
        if config_yaml.is_file():
            config = cdu_f.yamls.from_yaml(config_yaml)
            for k, (_, v) in config.items():
                self._cfg[k] = (self._cfg[k][0], v)

        # save configuration for reuse
        else:
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
            train: bool
    ):
        """Reset instance state after completing an epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        train : bool
            If True, training epoch. If False, validation epoch.

        Notes
        -----
        Must be implemented by child classes.
        Each instance has a `coords` attribute, a list of [fold, curve]
        indices.
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
        """Track resource consumption on CPU and CUDA device if available."""
        if self.pbar is None:
            return

        r_u, r_t = cdu.get_ram(scale=3)
        self._RAM = (
            self.pbar.add_task(
                "RAM use (GiB)", tabs=1, total=r_t, show_time=False)
            if self._RAM is None else self._RAM)
        self.pbar.update(self._RAM, completed=r_u)
        if self. device.type == "cpu":
            return

        g_u, g_r, g_t = utils.get_cuda_memory(self.device, scale=3)
        if self._GPU is None:
            self._GPU = self.pbar.add_task(
                "GPU use (GiB)", tabs=2, total=g_t, show_time=False)
            self._GPR = self.pbar.add_task(
                "GPU res (GiB)", tabs=2, total=g_t, show_time=False)

        self.pbar.update(self._RAM, completed=r_u)
        self.pbar.update(self._GPR, completed=g_r)

    def _initialize(
            self
    ):
        """Reinitialize trainable parameters."""
        self.model = self._cfg[self._M][0](**self._cfg[self._M][1])
        utils.set_mode(
            self.model, train=True, device=self.device, dtype=self.dtypes[0])
        self.criterion = self._cfg[self._C][0](**self._cfg[self._C][1])
        self.optimizer = self._cfg[self._O][0](
            params=self.model.parameters(), **self._cfg[self._O][1])
        self.scheduler = self._cfg[self._S][0](
            optimizer=self.optimizer, **self._cfg[self._S][1])

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
        utils.set_mode(
            self.model, train=train, device=self.device, dtype=self.dtypes[0])

        # loop over dataset once
        r_stats = []
        for sample, target in loader:
            # forward pass, backpropagation, optimization, and statistics
            output, loss, target = self._step(sample, target, train=train)
            r_stats += [[loss.item(), self._step_stats(output, target)]]
            self._track_memory()
            del output, loss, target

        # compute statistics and clean up after epoch
        agg_loss, agg_acc = np.mean(np.array(r_stats), axis=0)
        self._epoch_reset(epoch, train=train)

        # save data
        mode = "train" if train else "valid"
        data = [len(loader.dataset), epoch, mode, agg_loss, agg_acc]
        stats = pd.DataFrame(
            [self.names + self.coords + data], columns=self.COLS, index=[0])
        cdu_f.csvs.append_data(file=self.stat_csv, data=stats, index=False)
        return agg_loss, agg_acc

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        epochs: int,
        fold: int = 0,
        curve: int = 0,
        task_id: cdu.rp.TaskID = None,
        min_iter: int = 100,
        use_acc: bool = True
    ):
        """Train a pytorch model on a preconfigured train/test dataset split.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            Annotated dataset on which to train model.
        valid_dataset : torch.utils.data.Dataset
            Annotated dataset on which to validate model after each epoch.
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

        Other Parameters
        ----------------
        task_id : cdu.rp.TaskID, optional
            Index of epoch progress bar in instance attribute `pbar`.
            Defaults to None.
        min_iter : int, optional
            Minimum allowed iterations in an epoch. Useful if optimum batch
            size is much larger than dataset size.
        use_acc : bool, optional
            If True, consider increasing accuracy in addition to decreasing
            loss when saving model.
            Defaults to True.

        Notes
        -----
        `train_dataset` and `valid_dataset` should define a __getitem__ method
        that returns an iterable with at least two indices. The first index is
        used as input data while the second index serves as the target.
        """
        self.coords, epoch = [fold, curve], 0
        v_min, v_max = None, None if self.stat is None else self.stat[
            self.stat[self.COLS[6]] == "valid"][self.COLS[7:]].max().to_numpy()

        self._initialize()
        self._track_memory()
        if self.batch_size is None:  # simulate optimum batch size
            pair = train_dataset[0]
            self.batch_size = metrics.simulate_batch_size(
                self.model, sample=pair[0], device=self.device, target=pair[1],
                criterion=self.criterion, optimizer=self.optimizer,
                sample_dtype=self.dtypes[0], target_dtype=self.dtypes[1])
            self.batch_size = min(
                self.batch_size, len(train_dataset) // min_iter)

        # load model-specific checkpoint if available
        if self.curr_path.is_file() and self.stat is not None:
            stat = self.stat.loc[
                (self.stat[self.COLS[2]] == self.coords[0]) &
                (self.stat[self.COLS[3]] == self.coords[1])]

            # only load if there is a model with the same fold and curve index
            if not stat.empty:
                epoch = stat[self.COLS[5]].max() + 1
                if epoch < epochs:  # ... with an incomplete training loop
                    print(f"resuming fold {fold} curve {curve} epoch {epoch}")
                    utils.load(
                        self.curr_path, self.model, device=self.device,
                        load_dict={
                            self._O: self.optimizer, self._S: self.scheduler})
                else:  # skip otherwise
                    print(f"skipping fold {fold} curve {curve}")
                    return

        # prepare datasets
        train_loader = utils.get_dataloader(train_dataset, self.batch_size)
        valid_loader = utils.get_dataloader(valid_dataset, self.batch_size)

        # loop over full dataset per epoch
        for e in range(epoch, epochs):
            t_loss, t_acc = self._epoch(train_loader, train=True, epoch=e)
            v_loss, v_acc = self._epoch(valid_loader, train=False, epoch=e)

            # modify learning rate based on validation loss
            self.scheduler.step(v_loss)

            # save current checkpoint to resume if training pauses
            utils.save(self.curr_path, self.model, save_dict={
                self._O: self.optimizer, self._S: self.scheduler})

            # save model if peak validation performance
            check = {"curr_loss": v_loss, "best_loss": v_min}
            check.update(
                {"curr_acc": v_acc, "best_acc": v_max} if use_acc else {})
            if utils.is_better(**check):
                v_min, v_max = v_loss, v_acc
                message = f"Saving fold {fold} curve {curve} epoch {e}"
                message += f"\n    validation loss: {v_loss:.4e}"
                message += f"\n    validation metric {v_acc:.4e}"
                print(message)
                utils.save(self.peak_path, self.model)

            # update progress bar
            if self.pbar is not None:
                label = f"{len(train_loader)} batches of {self.batch_size}"
                self.pbar.start_task(task_id)
                self.pbar.update(task_id, label=label, completed=e + 1)

        self._make_plots()

        # remove within-loop values from memory
        del train_loader, valid_loader
        del self.model, self.criterion, self.optimizer, self.scheduler
        self.curr_path.unlink()
        self.curr_path.with_suffix(".pth").unlink()
        torch.cuda.empty_cache()

    def evaluate(
            self,
            eval_dataset: torch.utils.data.Dataset,
            batch_size: int = 10,
            task_id: cdu.rp.TaskID = None,
    ):
        """Inference on unlabeled data with trained model.

        Parameters
        ----------
        eval_dataset : torch.utils.data.Dataset
            Unlabeled dataset used for inference.
        batch_size : int
            Batch size used for inference.

        Yields
        ------
        int
            Index of first element in current batch along `eval_dataset`.
        np.ndarray
            Model output for current batch.

        Other Parameters
        ----------------
        task_id : cdu.rp.TaskID, optional
            Index of epoch progress bar in instance attribute `pbar`.
            Defaults to None.

        Notes
        -----
        `train_dataset` and `valid_dataset` should define a __getitem__ method
        that returns an iterable with at least two indices. The first index is
        used as input data while the second index serves as the target.
        """
        self._initialize()
        utils.load(self.peak_path, self.model, device=self.device)
        utils.set_mode(
            self.model, train=False, device=self.device, dtype=self.dtypes[0])
        eval_loader = utils.get_dataloader(
                eval_dataset, batch_size=batch_size, shuffle=False)
        if self.pbar is not None:
            self.pbar.start_task(task_id)
            self.pbar.update(task_id, total=len(eval_loader))

        for b, batch in enumerate(eval_loader):
            output, _, _ = utils.forward_pass(
                self.model, batch, device=self.device,
                sample_dtype=self.dtypes[0], target_dtype=self.dtypes[1])
            if self.pbar is not None:
                self.pbar.update(task_id, advance=1)

            yield b * batch_size, output.detach().cpu().numpy()
