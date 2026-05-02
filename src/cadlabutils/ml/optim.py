#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:00:00 2025
@author: ike
"""


# 1. Standard library imports
from pathlib import Path

# 2. Third-party library imports
import optuna
from optuna.trial import TrialState


def pull_trials(
        study_sqlite: Path,
        name: str,
        completed: bool = True
):
    """Extract hyperparameter combinations from an optuna study.

    Parameters
    ----------
    study_sqlite : Path
        Path to stored optuna study (sqlite).
    name : str
        Name of study within `study_sqlite`.
    completed : bool, optional
        If true, return only trials that ran to completion (success or pruned).

    Returns
    -------
    trials : list[optuna.Trial]
        Optuna trials.
    """
    study = optuna.load_study(
        study_name=name, storage=f"sqlite:///{study_sqlite}")
    trials = study.trials if not completed else [
        t for t in study.trials if t.value is not None]
    return trials


def resume_failed_trial(
        study: optuna.Study
):
    """Resume study by re-enqueueing recently failed trials.

    Parameters
    ----------
    study : optuna.Study
        Optuna study.

    Returns
    -------
    count : int
        Number of completed (success, pruned) trials in `study`.
    re_enqueue : bool
        If True, re-enqueued most recent trial in `study` to resume last failed
        hyperparameter combination.
    """
    count = sum(
        t.state in {TrialState.COMPLETE, TrialState.PRUNED}
        for t in study.trials)
    re_enqueue = False
    if len(study.trials) > 0 and study.trials[-1].state == TrialState.FAIL:
        study.enqueue_trial(study.trials[-1].params)
        re_enqueue = True

    return count, re_enqueue
