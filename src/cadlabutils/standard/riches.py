#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import time
import pandas as pd
import rich.progress as rp

from rich.text import Text
from rich.tree import Tree
from rich.table import Table
from rich.console import Console
from rich.traceback import install

from .console import elapsed_time


install(
    show_locals=True, width=120, suppress=["numpy", "pandas", "torch"])


def pbar(
        item,
        desc: str = "",
        tabs: int = 0
):
    """Generate a colorful progress bar in the terminal.

    Parameters
    ----------
    item
        Iterable to wrap in progress bar.
    desc : str, optional
        Description of the progress bar.
        Defaults to "".
    tabs : int, optional
        Number of tabs to offset left edge of progress bar.
        Defaults to 0.

    Returns
    -------
    bar
        Instantiated progress bar.
    """
    bar = rp.track(item, description=f"{" " * 4 * tabs}{desc}")
    return bar


def get_rich_table(
        df: pd.DataFrame
):
    """Convert any dataframe to a rich table.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to convert. May include multiindices.

    Returns
    -------
    table : Table
        Rich table.

    Examples
    --------
    >>> test_col = pd.MultiIndex.from_arrays(
    ...     [["A", "A", "B"], ["x", "y", "x"]], names=("group", "var"))
    >>> test_index = pd.MultiIndex.from_arrays(
    ...     [["one", "one", "two"], ["X", "Y", "X"]], names=("out", "in"))
    >>> test_df = pd.DataFrame(
    ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ...     index=test_index, columns=test_col)
    >>> test_table = get_rich_table(test_df)
    >>> Console(force_terminal=False).print(test_table)
    ┏━━━━━┳━━━━┳━━━┳━━━┳━━━┓
    ┃     ┃    ┃ A ┃ A ┃ B ┃
    ┃ out ┃ in ┃ x ┃ y ┃ x ┃
    ┡━━━━━╇━━━━╇━━━╇━━━╇━━━┩
    │ one │ X  │ 1 │ 2 │ 3 │
    │ one │ Y  │ 4 │ 5 │ 6 │
    │ two │ X  │ 7 │ 8 │ 9 │
    └─────┴────┴───┴───┴───┘
    """
    # Flatten column MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "\n".join([str(c) for c in col if c is not None])
            for col in df.columns]

    # Reset row MultiIndex if present
    df = df.reset_index(
        drop=False) if isinstance(df.index, pd.MultiIndex) else df
    table = Table(show_header=True, header_style="bold magenta")
    for col in df.columns:
        table.add_column(str(col))

    for _, row in df.iterrows():
        table.add_row(*[str(x) for x in row.tolist()])

    return table


def print_rich_tree(
        data,
        title: str = "tree",
        color: bool = True
):
    """Pretty-print a nested dictionary using rich's Tree with ├── and └──.

    Parameters
    ----------
    data : dict
        Dictionary to print. Values may be dict or simple types.
    title : str, optional
        Title of the tree root node.
        Defaults to "tree".
    color : bool, optional
        If True, print tree branches and leaf labels in color.
        Defaults to True.

    Examples
    --------
    >>> test_data = {
    ...     "lunch": {
    ...         "drink": ("water", "tea", "soda"),
    ...         "entreé": "enchiladas",
    ...     },
    ...     "dinner": {
    ...         "drink": ("water", "tea", "soda"),
    ...         "entreé": {
    ...             "steak": ("filet", "strip"),
    ...             "pasta": ("ragú", "pesto")
    ...         },
    ...     }
    ... }
    >>> print_rich_tree(test_data, title="menu", color=False)
    menu
    ├── lunch
    │   ├── drink: ('water', 'tea', 'soda')
    │   └── entreé: enchiladas
    └── dinner
        ├── drink: ('water', 'tea', 'soda')
        └── entreé
            ├── steak: ('filet', 'strip')
            └── pasta: ('ragú', 'pesto')
    """
    def _add_dict_to_tree(d, tree):
        for k, v in d.items():
            if isinstance(v, dict):
                branch = tree.add(f"[bold cyan]{k}[/]")
                _add_dict_to_tree(v, branch)
            else:
                tree.add(f"[green]{k}[/]: {v}")

    tree = Tree(f"[bold red1]{title}[/]")
    _add_dict_to_tree(data, tree)
    Console(force_terminal=color).print(tree)


class TimeSpeedColumn(rp.ProgressColumn):
    """Custom column: show elapsed time and speed, blank if show_time=False."""
    _NUL = "-:--:-- -.-- s/it"

    def render(self, task: rp.Task) -> Text:
        if not task.fields.get("show_time", True):
            return Text("")
        elif task.start_time is None:
            return Text(self._NUL, style="progress.elapsed")
        elif task.finished:
            return Text(
                task.fields.get("e_str", self._NUL), style="progress.elapsed")

        elapsed = time.perf_counter() - task.start_time
        speed = (elapsed / task.completed) if task.completed > 0 else 0.0
        task.fields["e_str"] = (
            self._NUL if elapsed == 0 else
            f"{elapsed_time(elapsed, is_elapsed=True)[-1]} {speed:5.2f} s/it")
        return Text(task.fields["e_str"], style="progress.elapsed")


class TreeBar(rp.Progress):
    TAB = "    "
    DWN = "|   "
    FRK = "├── "
    BTM = "└── "

    def __init__(
            self,
            **kwargs
    ):
        template = "[bold white]{task.fields[elbow]}"
        template += "[{task.fields[color]}]{task.fields[label]}"
        super(TreeBar, self).__init__(  # label
            rp.TextColumn(template, justify="left"),
            rp.BarColumn(bar_width=None),  # progress bar
            rp.MofNCompleteColumn(),  # shows X/Y
            TimeSpeedColumn(),
            # rp.TimeElapsedColumn(),  # total elapsed time
            **kwargs)
        self._ids, self._tabs = [], []

    def _update_tree(
            self
    ):
        for i, tabs in enumerate(self._tabs):
            pre, color = "", "green"
            if i < len(self._tabs) - 1 and self._tabs[i+1] > tabs:
                color = "bold cyan"

            for t in range(1, tabs):
                sub = self._tabs[i + 1:]
                if t in sub:
                    pre += self.DWN if min(
                        sub[:sub.index(t) + 1]) == t else self.TAB
                else:
                    pre += self.TAB

            if tabs > 0:
                sub = self._tabs[i + 1:]
                if tabs in sub:
                    pre += self.FRK if min(
                        sub[:sub.index(tabs) + 1]) >= tabs else self.BTM
                else:
                    pre += self.BTM

            self.update(
                self._ids[i], elbow=pre,
                color="bold red1" if tabs == 0 else color)

    def add_task(
            self,
            label: str,
            tabs: int | str = 0,
            **kwargs
    ):
        tabs = max(self._tabs) if tabs == "max" else tabs
        task_id = super(TreeBar, self).add_task(
            "", elbow="", label=label, color="green", **kwargs)
        self._ids += [task_id]
        self._tabs += [tabs]
        self._update_tree()
        return task_id

    def remove_task(
            self,
            task_id
    ):
        idx = self._ids.index(task_id)
        super(TreeBar, self).remove_task(task_id)
        self._ids.pop(idx)
        self._tabs.pop(idx)
        self._update_tree()
