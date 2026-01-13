#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


# 1. Standard library imports
from abc import ABC, abstractmethod
import time
from pathlib import Path

# 2. Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    ckpt_path : Path
        Path where model parameters are stored.
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
    scheduler_kwargs : dict, optional
        Keyword arguments passed to `scheduler` init.
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
    _RAM, _GPU = None, None
    _M, _C, _O, _S = "model", "criterion", "optimizer", "scheduler"
    COLS = [
        "trainer", "model", "name", "fold", "curve", "n", "b_s", "epoch",
        "mode", "t_sample", "loss", "acc"]

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
            pbar: cdu.TreeBar = cdu.classes.NullObject(),
    ):
        name = cdu.clean_name(Path(name)).stem
        out_dir = out_dir.joinpath("models")
        self.my_dir = out_dir.joinpath(model.__name__, name)
        self.test_dir = self.my_dir.joinpath("test")

        # set instance variables
        self.device = utils.get_device(gpu)
        self.dtypes = dtypes
        self.ckpt_path = self.my_dir.joinpath("ckpt")
        self.peak_path = self.my_dir.joinpath("peak")
        self.stat_csv = out_dir.joinpath("coretrainer_stats.csv")
        self.batch_size = None
        self.p_bar = pbar
        self.names = [self.__class__.__name__, model.__name__, name]
        self.coords = [0, 0]

        # prepare reset dict and initialize trainable parameters
        self._cfg = {
            self._M: (model, model_kwargs),
            self._C: (criterion, criterion_kwargs or {}),
            self._O: (optimizer, optimizer_kwargs or {}),
            self._S: (scheduler, scheduler_kwargs or {})}

        # load existing configuration
        config_yaml = self.my_dir.joinpath(f"{name}_config.yaml")
        if config_yaml.is_file():
            config = cdu_f.yamls.from_yaml(config_yaml)
            for k, (_, v) in config.items():
                self._cfg[k] = (self._cfg[k][0], v)

        # save configuration for reuse
        else:
            self.my_dir.mkdir(exist_ok=True, parents=True)
            self.test_dir.mkdir(exist_ok=True, parents=True)
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

    def _plot(
            self
    ):
        """Generate graphs at the end of training."""
        stats = self.pull_stats()[
            ["fold", "curve", "epoch", "mode", "loss", "acc"]]
        unique = sorted(stats["mode"].unique().tolist())

        # plot loss and accuracy per curve over epochs, aggregate across folds
        fig, axes = plt.subplots(
            nrows=2, ncols=1, figsize=(6, 12), sharex=True, sharey=False)
        for i, y in enumerate(["loss", "acc"]):
            sns.lineplot(
                data=stats, x="epoch", y=y, ax=axes[i], hue="mode", lw=3,
                style="curve", errorbar=("se", 2), legend=i == 0,
                hue_order=unique, palette=dict(zip(
                    unique, sns.color_palette("rocket", len(unique)))))
            cdu.style_ax(
                axes[i], x_label="epoch", y_label=y,
                y_ticks=None if i == 0 else (0, 0.5, 1))
            if i == 0:
                axes[0].legend(
                    frameon=False, prop={"size": 14, "weight": "bold"},
                    loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)

        plt.savefig(
            self.my_dir.joinpath(f"training.png"), dpi=300,
            bbox_inches="tight")
        plt.close(fig)

    def _track_memory(
            self
    ):
        """Track resource consumption on CPU and CUDA device if available."""
        if not isinstance(self.p_bar, cdu.TreeBar):
            return

        r_u, r_t = cdu.get_ram(scale=3)
        if self._RAM is None:
            self._RAM = self.p_bar.add_task(
                    "RAM GiB", tabs=0, total=r_t, show_time=False)

        self.p_bar.update(self._RAM, completed=r_u)
        if self.device.type != "cpu":
            g_u, g_r, g_t = utils.get_cuda_memory(self.device, scale=3)
            if self._GPU is None:
                self._GPU = self.p_bar.add_task(
                    "GPU (GiB)", tabs=0, total=g_t, show_time=False)

            self.p_bar.update(self._GPU, completed=g_r)

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

    def _clean_up(
            self,
            task_id: cdu.rp.TaskID = None
    ):
        """Clean up after inference.

        Parameters
        ----------
        task_id : cdu.rp.TaskID, optional
            Index of loop progress bar in instance attribute `pbar`.
            Defaults to None.
        """
        del self.model, self.criterion, self.optimizer, self.scheduler
        torch.cuda.empty_cache()
        self._plot()
        for f in self.ckpt_path.parent.glob(f"{self.ckpt_path.name}*"):
            f.unlink()

        if self._RAM is not None:
            self.p_bar.remove_task(self._RAM)
        if self._GPU is not None:
            self.p_bar.remove_task(self._GPU)
        if task_id is not None:
            self.p_bar.stop_task(task_id)

        self._RAM, self._GPU = None, None

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
        custom logic to account for different input and output structures.
        Any such function must:
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
            delta = time.time()
            output, loss, target = self._step(sample, target, train=train)
            delta = (time.time() - delta) / sample.size(dim=0)
            r_stats += [[delta, loss.item(), self._step_stats(output, target)]]
            self._track_memory()
            del output, loss, target

        # compute statistics and clean up after epoch
        agg_time, agg_loss, agg_acc = np.mean(np.array(r_stats), axis=0)
        self._epoch_reset(epoch, train=train)

        # save data
        mode = "train" if train else "valid"
        data = [
            len(loader.dataset), self.batch_size, epoch, mode, agg_time,
            agg_loss, agg_acc]
        stats = pd.DataFrame(
            [self.names + self.coords + data], columns=self.COLS, index=[0])
        cdu_f.csvs.append_data(file=self.stat_csv, data=stats, index=False)
        return agg_loss, agg_acc

    def pull_stats(
            self,
            cols: str | list[str] = None
    ):
        """Pull training statistics related to current model.

        Parameters
        ----------
        cols : str | list[str], optional
            Names of statistic columns to pull. Columns are those defined in
            class attribute `COLS`.
            Defaults to None, in which case all columns are returned.

        Returns
        -------
        stats : pd.DataFrame | None
            Stored statistics of prior epochs for current model and paradigm.
            None if no such statistics are available.
        """
        stats = None
        if self.stat_csv.is_file():
            stats = pd.read_csv(self.stat_csv).query(" and ".join([
                f"{c} == '{n}'" for c, n in zip(self.COLS[:3], self.names)]))
            stats = stats if cols is None else stats[cols]
            stats = None if stats.empty else stats

        return stats

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        epochs: int,
        fold: int = 0,
        curve: int = 0,
        task_id: cdu.rp.TaskID = None,
        min_iter: int = 100,
        use_acc: bool = True,
        save_best: bool = True,
        trial: optuna.Trial = None
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
        save_best : bool, optional
            If True, save best model as dedicated file.
            Defaults to True.
        trial : optuna.Trial, optional
            Optuna trial object used for hyperparameter optimization.

        Notes
        -----
        `train_dataset` and `valid_dataset` should define a __getitem__ method
        that returns an iterable with at least two indices. The first index is
        used as input data while the second index serves as the target.
        """
        self.coords, epoch = [fold, curve], 0
        globe_loss, globe_acc = float("inf"), float("-inf")
        local_loss, local_acc = float("inf"), float("-inf")
        stats = self.pull_stats()
        if stats is not None:
            self.batch_size = int(stats.iloc[0, 6])
            stats = stats.query(f"{self.COLS[8]} == 'valid'")
            globe_loss, globe_acc = (
                stats[self.COLS[-2]].min(), stats[self.COLS[-1]].max())
            print(f"global: l{globe_loss:.2e} a{globe_acc:.2%}")
            stats = stats.query(f"{self.COLS[3]} == @fold")
            if not stats.empty:
                local_loss, local_acc = (
                    stats[self.COLS[-2]].min(), stats[self.COLS[-1]].max())
                print(f"local: l{local_loss:.2e} a{local_acc:.2%}")
                print(self.batch_size)

            stats = stats.query(f"{self.COLS[4]} == @curve")

        self._initialize()
        self._track_memory()

        # simulate optimum batch size
        if self.batch_size is None:
            pair = train_dataset[0]
            self.batch_size = metrics.simulate_batch_size(
                self.model, sample=pair[0], device=self.device, target=pair[1],
                criterion=self.criterion, optimizer=self.optimizer,
                sample_dtype=self.dtypes[0], target_dtype=self.dtypes[1])
            self.batch_size = min(
                self.batch_size, len(train_dataset) // min_iter)

        # load model-specific checkpoint if available
        ckpt = self.ckpt_path.with_suffix(".safetensors")
        if ckpt.is_file() and stats is not None and not stats.empty:
            epoch = stats[self.COLS[7]].max() + 1
            print(epoch, epochs)
            import sys; sys.exit()
            if epoch >= epochs:
                return

            print(f"resuming fold {fold} curve {curve} epoch {epoch}")
            utils.load(
                self.ckpt_path, self.model, device=self.device, load_dict={
                    self._O: self.optimizer, self._S: self.scheduler})
        import sys; sys.exit()
        # prepare datasets
        train_loader = utils.get_dataloader(train_dataset, self.batch_size)
        valid_loader = utils.get_dataloader(valid_dataset, self.batch_size)

        label = f"{len(train_loader)} x {self.batch_size}"
        self.p_bar.start_task(task_id)
        self.p_bar.update(task_id, label=label, completed=epoch)

        # loop over full dataset per epoch
        for e in range(epoch, epochs):
            t_loss, t_acc = self._epoch(train_loader, train=True, epoch=e)
            v_loss, v_acc = self._epoch(valid_loader, train=False, epoch=e)

            # modify learning rate based on validation loss, save checkpoint
            self.scheduler.step(v_loss)
            utils.save(self.ckpt_path, self.model, save_dict={
                self._O: self.optimizer, self._S: self.scheduler})

            # save model if peak validation performance
            local_check = utils.is_better(
                (v_loss, local_loss), (v_acc, local_acc) if use_acc else None)
            globe_check = utils.is_better(
                (v_loss, globe_loss), (v_acc, globe_acc) if use_acc else None)
            if save_best and local_check:
                local_loss, local_acc = v_loss, v_acc
                utils.save(self.my_dir.joinpath(f"fold {fold}"), self.model)
                if globe_check:
                    globe_loss, globe_acc = v_loss, v_acc
                    utils.save(self.peak_path, self.model)
                    print(f"{self.coords + [e]}: l{v_loss:.2e} a{v_acc:.2%}")

            self.p_bar.update(task_id, completed=e + 1)

            # optional optuna hyperparameter optimization
            if trial is not None:
                trial.report(v_acc if use_acc else v_loss, step=e)
                if trial.should_prune():
                    del train_loader, valid_loader
                    self._clean_up(task_id)
                    raise optuna.TrialPruned()

        # remove within-loop values from memory
        del train_loader, valid_loader
        self._clean_up(task_id)

    def evaluate(
            self,
            eval_dataset: torch.utils.data.Dataset,
            task_id: cdu.rp.TaskID = None,
            in_idx: int | str = slice(None),
            logits: bool = True,
            fold: int = None
    ):
        """Inference on unlabeled data with trained model.

        Parameters
        ----------
        eval_dataset : torch.utils.data.Dataset
            Unlabeled dataset used for inference.

        Yields
        ------
        int
            Index of first element in current batch along `eval_dataset`.
        np.ndarray
            Model output for current batch.

        Other Parameters
        ----------------
        task_id : cdu.rp.TaskID, optional
            Index of batch progress bar in instance attribute `pbar`.
            Defaults to None.
        in_idx : int | str, optional
            Index of each sample in `eval_dataset` to use for inference.
            Defaults to `slice(None)`, in which case the entirety of each
            sample is used as model input.
        logits : bool, optional
            If True, values in `output` are interpreted as raw logits.
            Defaults to True.

        Notes
        -----
        `train_dataset` and `valid_dataset` should define a __getitem__ method
        that returns an iterable with at least two indices. The first index is
        used as input data while the second index serves as the target.
        """
        model_path = self.peak_path if fold is None else self.my_dir.joinpath(
            f"fold {fold}")
        batch_size = int(self.pull_stats().iloc[0, 6])
        self._initialize()
        self._track_memory()
        utils.load(model_path, self.model, device=self.device)
        utils.set_mode(
            self.model, train=False, device=self.device, dtype=self.dtypes[0])
        eval_loader = utils.get_dataloader(
            eval_dataset, batch_size=batch_size, shuffle=False)
        self.p_bar.update(task_id, total=len(eval_loader), start=True)
        for b, batch in enumerate(eval_loader):
            output, _, _ = utils.forward_pass(
                self.model, batch[in_idx], device=self.device,
                sample_dtype=self.dtypes[0], target_dtype=self.dtypes[1])
            output = F.softmax(output, dim=1) if logits else output
            self.p_bar.update(task_id, completed=b + 1)
            yield b * batch_size, output.detach().cpu().numpy()

        del eval_loader
        self._clean_up(task_id)
