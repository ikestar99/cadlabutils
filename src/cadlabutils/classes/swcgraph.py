#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 8 09:00:00 2025
@author: Ike
"""


# 1. Standard library imports
from io import StringIO
from pathlib import Path

# 2. Third-party library imports
import numpy as np
import pandas as pd

# 3. Local application / relative imports
import cadlabutils as cdu
import cadlabutils.files as cdu_f


class SWCGraph:
    """Class to represent neuronal morphologies stored in swc format.

    Class Attributes
    ----------------
    RESOLUTION : dict
        Default resolution info for all neuronal morphologies.
    C_N : str
        Name of column containing node numerical identifier.
    C_T : str
        Name of column containing node class numerical identifier.
    C_XYZ : list
        Names of columns containing node coordinates.
    C_ZYX : list
        Reversed names of columns containing node coordinates.
    C_R : str
        Name of column containing node radius.
    C_U : str
        Name of column containing parent node numerical identifier.
    C_ALL : list
        All column names in standard order. No

    Attributes
    ----------
    data : pd.DataFrame
        Neuronal morphology. Has 7 columns specified by class attributes.
    resolution : dict
        Resolution of each spatial dimension in microns/pixel. Keys correspond
        to coordinate columns in C_XYZ attribute.
    coords : np.ndarray
        Coordinates of all nodes. Of shape (nodes, z y x).
    center : np.ndarray
        (z, y, x) of soma if present, otherwise center of mass of all nodes.
    components : np.ndarray
        Label of connected component to which each node belongs. Of shape
        (nodes,).
    offset : np.ndarray
        Minimum coordinates of furthest node in each axis, with padding.

    Parameters
    ----------
    data : pd.DataFrame):
        Formatted morphology data. Each row corresponds to a node and contains
        ordered columns [node id, type, x, y, z, radius, parent node id].
    resolution (dict, optional):
        Pixel resolution, in microns, of each spatial dimension. Used to
        convert coordinates in microns to coordinates in pixels. Key is the
        dimension ("z", "y", "x") and value is resolution.
        Defaults to None, in which case coordinates are not scaled.
    """
    RESOLUTION = {"x": 1, "y": 1, "z": 1}
    C_N = "node"
    C_T = "type"
    C_XYZ = ["x", "y", "z"]
    C_ZYX = ["z", "y", "x"]
    C_R = "radius"
    C_U = "parent"
    C_ALL = [C_N, C_T] + C_XYZ + [C_R, C_U]
    COLORS = ["black", "#0C7BDC", "#FFC20A"]  # soma axon dendrite

    def __init__(
            self,
            data: pd.DataFrame,
            resolution: dict[str: float] = None
    ):
        # standardize column names
        self.data = data
        self.data.columns = self.C_ALL

        # convert node coordinates based on imaging resolution
        self.resolution = self.RESOLUTION.copy()
        if resolution is not None:
            self.update_resolution(resolution)

    def __len__(
            self
    ):
        """Get number of nodes in morphology.

        Returns
        -------
        int
            Number of nodes in morphology.
        """
        return self.data.shape[0]

    def __getitem__(
            self,
            item: dict[str: list[float] | tuple[float]]
    ):
        """Filter column values and return result as new instance.

        Parameters
        ----------
        item : dict
            key : str
                Column name on which to perform filtering.
            value : list | tuple[float, float]
                List defines values to retain after filtering. Tuple of form
                (lower bound, upper bound) defines numerical values to retain
                after filtering.

        Returns
        -------
        SWCGraph
            Instance with filtered data.
        """
        # mask data by column values
        subset = None
        for k, v in item.items():
            mask = (
                self.data[k].isin(v) if isinstance(v, list) else
                self.data[k].between(v[0], v[1]))
            subset = mask if subset is None else (subset & mask)

        # filter data based on mask
        subset = self.data.loc[subset].copy()
        if subset.shape[0] == 0:
            raise ValueError("No nodes found with input criteria")

        # return new filtered instance with updated attributes
        subset = SWCGraph(subset)
        subset.resolution = self.resolution
        return subset

    def __getattr__(
            self,
            name: str
    ):
        """Get up-to-date attribute values.

        Parameters
        ----------
        name : str
            Attribute to return.

        Returns
        -------
        object
            Attribute.

        Raises
        ------
        AttributeError:
            Attribute not defined.
        """
        match name:
            case n if n in self.C_ALL:
                attribute = self.data[name].copy().to_numpy()
            case "center":
                coords, mask = self.coords, self.type == 1
                attribute = coords[np.argmax(mask)] if np.any(mask) else np.mean(
                    coords, axis=0)
            case "components":
                node, parent = self.node, self.parent
                parent[parent == 1] = -1
                uf = cdu.classes.UnionFind(node.size)
                for i, n in enumerate(node):
                    if parent[i] != -1:
                        uf.union(np.argmax(node == parent[i]), i)

                attribute = uf.update()[0]
            case "coords":
                attribute = np.round(
                    self.data[self.C_ZYX].copy().to_numpy()).astype(int)
            case _:
                raise AttributeError(f"Skeleton has no attribute named {name}")

        return attribute

    @classmethod
    def from_file(
            cls,
            skeleton_swc: Path,
            resolution: dict[str: float] = None
    ):
        """Instantiate from an existing swc file.

        Parameters
        ----------
        skeleton_swc : Path
            Path to skeleton representation of neuronal morphology (swc).
        resolution : dict, optional
            Passed to __init__ constructor.
            Defaults to None.

        Returns
        -------
        SWCGraph
            Skeleton instance loaded from file.

        Notes
        -----
        There is prominent intergroup variability in the naming and saving swc
        files. This class assumes:
        -   swc file is formatted as a csv file with " " or "," separators.
        -   Each row has 7 numerical columns in a standard order:
            node id, type, x/y/z coordinates, node radius, and parent node id.
        -   The first relevant data row contains numeric entries only.
        """
        data = pd.read_csv(
            skeleton_swc, sep=r"[, ]+", engine="python",
            skiprows=cdu_f.csvs.find_first_row(skeleton_swc), header=None)
        return cls(data=data.iloc[:, :7], resolution=resolution)

    @classmethod
    def from_text(
            cls,
            raw_text: str,
            resolution: dict[str: float] = None
    ):
        """Instantiate from a csv-like raw text string.

        Parameters
        ----------
        raw_text : str
            Entire contents of csv file as string.
        resolution : dict, optional
            Passed to __init__ constructor.
            Defaults to None.

        Returns
        -------
        SWCGraph
            Skeleton instance loaded from text data.
        """
        # load swc file
        data = pd.read_csv(
            StringIO(raw_text), sep=r"[, ]+", engine="python",
            skiprows=cdu_f.csvs.find_first_row(raw_text), header=None)
        return cls(data=data.iloc[:, :7], resolution=resolution)

    @classmethod
    def from_coordinates(
            cls,
            coords: np.ndarray,
            parents: np.ndarray,
            nodes: np.ndarray = None,
            types: np.ndarray = None,
            radii: np.ndarray = None,
            resolution: dict[str: float] = None
    ):
        """Instantiate from node coordinates and known parent ids.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates of each node in pixels, unless resolution is specified.
            Of shape (nodes, 3) with the second axis ordered as (z, y, x).
        parents : np.ndarray
            Parent node id of each node. Of shape (nodes,).
        nodes : np.ndarray, optional
            Node ids. Of shape (nodes,).
            Defaults to None, in which case node id is inferred from index.
        types : np.ndarray, optional
            Node types. Of shape (nodes,). 1 soma, 2 axon, 3 dendrite.
            Defaults to None, in which case node types default to 1.
        radii : np.ndarray, optional
            Node radii. Of shape (nodes,).
            Defaults to None, in which case node radii default to 1.
        resolution : dict, optional
            Passed to __init__ constructor.
            Defaults to None.

        Returns
        -------
        SWCGraph
            Skeleton instance assembled from components.
        """
        # fill missing columns with default values
        nodes = (nodes or np.arange(coords.shape[0]))[:, None]
        types = (types or np.ones(nodes.shape[0]))[:, None]
        radii = (radii or np.ones(nodes.shape[0]))[:, None]

        # stack data into dataframe
        data = pd.DataFrame(np.concatenate(
            [nodes, types, coords[..., ::-1], radii, parents[..., None]],
            axis=-1))
        return cls(data=data, resolution=resolution)

    def save(
            self,
            file_swc: Path,
            microns: bool
    ):
        """Save stored data as an .swc file.

        Parameters
        ----------
        file_swc : Path
            Path to save stored data (swc).
        microns : bool
            If True, convert coordinates to microns before saving.

        Returns
        -------
        SWCGraph
            Instance.

        Notes
        -----
        Save will convert coordinate values to microns before saving, and back
        to pixels after.
        """
        # convert coordinate values from pixels to microns
        if microns:
            res = self.resolution.copy()
            self.update_resolution(self.RESOLUTION.copy())

        # save data in microns
        save = self.data.copy()
        save.columns = ",-".join(self.C_ALL).split("-")
        save.to_csv(file_swc, sep=" ", header=True, index=False)

        # restore coordinate values from microns to pixels
        if microns:
            self.update_resolution(res)

        return self

    def update_resolution(
            self,
            resolution: dict[str: float]
    ):
        """Update resolution of coordinate columns.

        Parameters
        ----------
        resolution : dict
            Pixel resolution, in microns, of each spatial dimension. Used to
            convert coordinates in microns to coordinates in pixels. Key is the
            dimension ("z", "y", "x") and value is resolution.

        Returns
        -------
        SWCGraph
            Instance with updated spatial resolution.
        """
        c_new = [k for k in resolution if k in self.resolution]
        c_old = [k for k in self.resolution if k in c_new]

        # undo prior resolution operation on specified columns
        self.data[c_old] = self.data[c_old].mul(
                pd.Series({k: self.resolution[k] for k in c_old}))

        # apply new resolution scale factor(s)
        self.data[c_new] = self.data[c_new].div(pd.Series(resolution))
        self.resolution = {
            k: resolution.get(k, v) for k, v in self.resolution.items()}
        return self

    def get_bounds(
            self,
            pad: int = 0
    ):
        """Get coordinates of bounding box that includes all nodes.

        Parameters
        ----------
        pad : int, optional
            Symmetric padding on all faces of the extracted bounding box.
            Defaults to 0.

        Returns
        -------
        min_idx : np.ndarray
            Minimum integer coordinates across all three axes. (z, y, x).
        max_idx : np.ndarray
            Maximum integer coordinates across all three axes. (z, y, x).
        """
        min_idx = np.min(self.coords, axis=0) - pad
        min_idx[min_idx < 0] = 0
        max_idx = np.max(self.coords, axis=0) + pad
        return min_idx, max_idx

    def translate(
            self,
            offset: np.ndarray
    ):
        """Translate stored coordinates by specified offset in pixels.

        Parameters
        ----------
        offset : np.ndarray
            Number of pixels to translate nodes per axis. (z, y, x).

        Returns
        -------
        SWCGraph
            Instance with translated coordinates.
        """
        self.data[self.C_ZYX] = self.data[self.C_ZYX] - offset
        return self

    def get_distances(
            self,
            center: np.ndarray = None
    ):
        """Return the distance between each node and reference point.

        Parameters
        ----------
        center : np.ndarray, optional
            Point against which to compute distances. (z, y, x).
            Defaults to None, in which case center point is soma.

        Returns
        -------
        np.ndarray
            Euclidean distance between each node and reference point. (nodes,).
        """
        distances = (self.coords - np.array(center or self.center)[None]) ** 2
        return np.sqrt(np.sum(distances, axis=1))

    # def get_mask(
    #         self,
    #         chunk: np.array,
    #         start: np.ndarray,
    #         d_max: float,
    #         invert: bool,
    #         collapse: str = None,
    #         downsample: int = None
    # ):
    #     """
    #     Apply topology-preserving fast marching algorithm to convert raw data
    #     stack into a volumetric foreground mask using skeleton nodes as seed
    #     points.
    #
    #     NOTE: algorithm expects input images with bright foreground and dark
    #     background -- use invert argument appropriately.
    #
    #     NOTE: to generate a mask from a 2d projection, chunk arg should have
    #     shape 1 along the projected axis and start arg should indicate an
    #     initial coordinate of 0 along this axis.
    #
    #     Args:
    #         chunk (np.ndarray):
    #             Raw data stack. Of shape (z, y, x) and dtype np.uint8.
    #         start (np.ndarray):
    #             Coordinate of chunk[0, 0, 0] in larger image stack. Used to
    #             select nodes found in the same subvolume. Of shape (z, y, x).
    #         d_max (float):
    #             Cutoff distance during fast march wave propagation. Passed to
    #             arealize.utils.fast_march function call.
    #         invert (bool):
    #             If True, invert raw image such that foreground and background
    #             pixels swap magnitudes.
    #         collapse (str, optional):
    #             Axis to collapse if masking a stack projection. "z", "y", "x".
    #             Defaults to None, in which case axis is not collapsed.
    #         downsample (int, optional):
    #             Use every Nth node to generate volumetric labels.
    #             Defaults to None, in which case all nodes are used.
    #
    #     Returns:
    #         (tuple):
    #             Contains the following three items:
    #             -   (np.ndarray):
    #                     Volumetric foreground mask. Same shape as chunk.
    #             -   (np.ndarray):
    #                     Foreground distance map. Same shape as chunk.
    #             -   (np.ndarray):
    #                     Foreground arrival time map. Same shape as chunk.
    #     """
    #     from ..marching import fast_march
    #     # account for missing axes
    #     chunk = (
    #         np.expand_dims(chunk, axis=self.C_ZYX.index(collapse))
    #         if collapse is not None else chunk)
    #
    #     # subset relevant nodes
    #     item = {
    #         k: (start[i], start[i] + chunk.shape[i])
    #         for i, k in enumerate(self.C_ZYX) if collapse != k}
    #
    #     # translate coordinates to match chunk indices, flatten unused axes
    #     seeds = self[item]
    #     seeds = seeds.coords[np.argsort(seeds.get_distances())] - start[None]
    #     seeds = seeds * np.array([[(k in item) for k in self.C_ZYX]])
    #     seeds = seeds if downsample is None else seeds[::downsample]
    #
    #     # apply fast march algorithm
    #     _, mask, distances, times = fast_march(
    #         255 - chunk if invert else chunk, seeds=seeds, d_max=d_max,
    #         aniso=[self.resolution[k] for k in self.C_ZYX])
    #     return (255 * (mask != 0)).astype(np.uint8), distances, times


