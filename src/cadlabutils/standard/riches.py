#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


from rich.tree import Tree
from rich.console import Console
from rich.progress import track, ProgressBar


def pbar(
        item,
        desc: str = "",
        tabs: int = 0
):
    """Generate a colorful progress bar in the terminal.

    Parameters
    ----------
    item : iterable
        Iterable to wrap in progress bar.
    desc : str, optional
        Description of the progress bar.
        Defaults to "".
    tabs : int, optional
        Number of tabs to offset left edge of progress bar.
        Defaults to 0.

    Returns
    -------
    ProgressBar
        Instantiated progress bar.
    """
    return track(item, description=f"{" " * 4 * tabs}{desc}")


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
