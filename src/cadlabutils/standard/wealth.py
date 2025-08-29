#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


from rich.tree import Tree
from rich.console import Console


def print_rich_tree(d, title="tree", color=True):
    """
    Pretty-print a nested dictionary using rich's Tree with ├── and └──.

    Parameters
    ----------
    d : dict
        Dictionary to print. Values may be dict or simple types.
    title : str, optional
        Title of the tree root node.
        Defaults to "tree".

    Examples
    --------
    >>> test_data = {
    ...     "lunch": {
    ...         "drink": ("water", "tea", "soda"),
    ...         "entreé": "enchiladas",
    ...     },
    ...     "dinner": {
    ...         "drink": ("water", "tea", "soda"),
    ...         "entreé": {"steak": ["filet", "strip"]},
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
            └── steak: ['filet', 'strip']
    """
    def _add_dict_to_tree(d, tree):
        for k, v in d.items():
            if isinstance(v, dict):
                branch = tree.add(f"[bold cyan]{k}[/]")
                _add_dict_to_tree(v, branch)
            else:
                tree.add(f"[green]{k}[/]: {v}")

    tree = Tree(f"[bold red1]{title}[/]")
    _add_dict_to_tree(d, tree)
    Console(force_terminal=color).print(tree)
