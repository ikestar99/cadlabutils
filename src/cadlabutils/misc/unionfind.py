#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import numpy as np
import scipy.spatial as ssp

from scipy.spatial.distance import cdist


class UnionFind:
    """
    UnionFind is a data structure class built to separate disjoint sets of
    connected nodes from a larger dataset. An index represents a single node in
    the full dataset. The value of instance attributes at that index correspond
    to properties of the related node, such as the parent node or number of
    child nodes at each root.

    ---------------------------------------------------------------------------
    Attributes:
        _root (list):
            Mapping to index of root node unique to each connected component.
            len(root) = number of nodes in full dataset.
        _prev (list):
            Mapping to index of direct upstream node. Value is initially set to
            -1 at every index and updated for each child node. Root nodes
            retain -1 value after all updates. Same length as root attribute.
        _rank (list):
            Number of child nodes connected directly or indirectly to root
            node. Same length as root attribute.
    """
    def __init__(
            self,
            n: int
    ):
        """
        Initialize union find data structure.

        Args:
            n (int):
                Number of nodes in full dataset.
        """
        self._root = list(range(n))
        self._prev = [-1] * n
        self._rank = [0] * n

    def find(
            self,
            x: int
    ):
        """
        Find root node of component that includes queried node. Recursive call
        that queries the root of current node until reaching a symmetry where
        a node is its own root.

        Args:
            x (int):
                Index of node for which to find root.

        Returns;
            (int):
                Index of root node of component that includes queried node.
        """
        if self._root[x] != x:
            self._root[x] = self.find(self._root[x])

        return self._root[x]

    def union(
            self,
            x: int,
            y: int
    ):
        """
        Join two nodes as parent/child. Root node after union call is the root
        with the greater number of child nodes, or the root of the first node
        if both have an equal number of child nodes.

        Args:
            x (int):
                Index of first node to join.
            y (int):
                Index of second node to join.
        """
        root_x, root_y = self.find(x), self.find(y)
        prev_x, prev_y = self._prev[x], self._prev[y]
        if root_x != root_y:
            if self._rank[root_x] < self._rank[root_y]:
                self._root[root_x] = root_y
                self._prev[x] = y if prev_x == -1 else prev_x
                self._rank[root_y] += 1
            else:
                self._root[root_y] = root_x
                self._prev[y] = x if prev_y == -1 else prev_y
                self._rank[root_x] += 1

    def update(
            self
    ):
        """
        Update properties of all nodes in dataset. Preferred means by which to
        access instance attributes as union function call may not update all
        other points with each addition.

        Returns:
            (tuple):
                Contains three items, all of dtype int:
                -   0 (np.ndarray):
                        Root node of each node in dataset. All nodes with the
                        same root are part of a unique connected component.
                -   1 (np.ndarray):
                        Parent node of each node in dataset. All root nodes
                        have a value of -1.
                -   2 (np.ndarray):
                        Number of child nodes connected to each root node. All
                        non-root nodes have a value of 0
        """
        for i in range(len(self._root)):
            _ = self.find(i)

        return np.array(self._root), np.array(self._prev), np.array(self._rank)


def find_neighbors(
    coordinates: np.ndarray
):
    """
    Apply cKDTree to identify indices of neighboring points. Neighboring nodes
    are directly connected by faces, vertices, or edges -- approximately within
    a radius equal to the magnitude of the vector <1, ..., coordinates.ndim>.

    Args:
        coordinates (np.ndarray):
            Input array of point coordinates. Of shape (n_points, n_axes).

    Returns:
        (np.ndarray):
            Array of lists with the same length as coordinates arg. Each index
            contains a list of indices of all points connected to the current
            point, inclusive of the queried point itself.
    """
    tree = ssp.cKDTree(coordinates)
    neighbors = np.array(
        tree.query_ball_tree(tree, r=coordinates.shape[-1] + 0.1), dtype=list)
    return neighbors


def find_components(
        coordinates: np.ndarray,
        center: np.ndarray
):
    """
    Identify connected components from array of node coordinates and assign
    parent/child relationships between adjacent nodes. Will call find_neighbors
    to identify indices of neighboring points. Root node of each connected
    component is node closest to center point.

    Args:
        coordinates (np.ndarray):
            Input array of point coordinates. Of shape (n_points, n_axes).
        center (np.ndarray):
            Reference point. Of shape (n_axes,).

    Returns:
        (tuple):
            Contains 4 items, all 1d arrays of same length as number of nodes:
            -   0 (np.ndarray):
                    Label of connected component to which each node belongs.
            -   1 (np.ndarray):
                    Parent node index of each node.
            -   2 (np.ndarray):
                    Number of child nodes connected to each root node.
            -   3 (np.ndarray):
                    Euclidean distance of each node from center point.
    """
    uf = UnionFind(coordinates.shape[0])
    neighbors = find_neighbors(coordinates)
    distances = cdist(center[None], coordinates).flatten()
    for x in np.argsort(distances):
        for n in neighbors[x]:
            if n != x:
                uf.union(x, n)

    return *uf.update(), distances
