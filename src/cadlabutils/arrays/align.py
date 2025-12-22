#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:06:21 2021
@author: ike
"""


# 2. Third-party library imports
import numpy as np
from pystackreg import StackReg
import skimage.transform as skt


# Instantiate registration machinery from pystackreg.
T_REG = {
    "translation": StackReg(StackReg.TRANSLATION),
    "rigid": StackReg(StackReg.RIGID_BODY),
    "rotation": StackReg(StackReg.SCALED_ROTATION),
    "affine": StackReg(StackReg.AFFINE),
    "bilinear": StackReg(StackReg.BILINEAR)
}


def register(
        ref: np.ndarray,
        mov: np.ndarray,
        mode: str
):
    """Register an image to a static reference.

    Parameters
    ----------
    ref : np.ndarray
        2D static reference image.
    mov : np.ndarray
        Image with same shape as `ref`. Registration maps mov onto ref.
    mode : str
        Constraints of registration paradigm:
        - "translation" --> translation in X/Y directions.
        - "rigid" --> rigid transformations.
        - "rotation" --> rotation and dilation.
        - "affine" --> affine transformation.
        - "bilinear" --> bilinear transformation.

    Returns
    -------
    matrix : np.ndarray
        3x3 transformation matrix mapping `mov` onto `ref`.

    Examples
    --------
    Rigid body registration
    >>> t_ref = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
    >>> t_mov = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 1]])
    >>> register(t_ref, t_mov, mode="rigid")
    array([[ 1., -0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    """
    matrix = T_REG[mode].register(ref, mov)
    return matrix


def transform(
        mov: np.ndarray,
        matrix: np.ndarray,
        order: int,
        cval: float = 0.0
):
    """Transform an image using a known transformation matrix.

    Parameters
    ----------
    mov : np.ndarray
        2D image to be transformed.
    matrix : np.ndarray
        3x3 transformation matrix with which to transform mov.
    order : int
        Interpolation paradigm for transformation. Valid values are integers on
        range [0 (nearest-neighbor), 5 (bi-quintic)].
    cval : float, optional
        Fill value for warped pixels with no equivalent in `mov`.
        Defaults to 0.

    Returns
    -------
    moved : np.ndarray
        Transformed image.
    """
    moved = skt.warp(
        mov, matrix, order=order, mode="constant", preserve_range=True,
        cval=cval).astype(mov.dtype)
    return moved


def align_image(
        ref: np.ndarray,
        mov: np.ndarray,
        cycles: int = 1,
        mode: str = "rigid",
        order: int = 0,
        cval: float = 0.0
):
    """Register and transform an image to a static reference.

    Parameters
    ----------
    ref : np.ndarray
        2D static reference image.
    mov : np.ndarray
        2D image to align. Must have same shape as `ref`.
    cycles : int, optional
        Number of times to repeat alignment.
        Defaults to 1.
    mode : str, optional
        Constraints of registration paradigm:
        - "translation" --> translation in X/Y directions.
        - "rigid" --> rigid transformations. Default option.
        - "rotation" --> rotation and dilation.
        - "affine" --> affine transformation.
        - "bilinear" --> bilinear transformation.
    order : int, optional
        Interpolation paradigm for transformation. Valid values are integers on
        range [0 (nearest-neighbor), 5 (bi-quintic)].
        Defaults to 0.
    cval : float, optional
        Fill value for warped pixels with no equivalent in `mov`.
        Defaults to 0.0.

    Returns
    -------
    mov : np.ndarray
        Transformed image.
    matrix : np.ndarray
        3x3 transformation matrix mapping final `mov` output onto `ref`.

    Examples
    --------
    """
    for _ in range(cycles):
        matrix = register(ref=ref, mov=mov, mode=mode)
        mov = transform(mov=mov, matrix=matrix, order=order, cval=cval)

    return mov, matrix


def align_stack(
        ref: np.ndarray,
        mov_stack: np.ndarray,
        static_mov: np.ndarray = None,
        cycles: int = 1,
        mode: str = "rigid",
        order: int = 0,
        cval: float = 0.0
):
    """Register and transform an image to a static reference.

    Parameters
    ----------
    ref : np.ndarray
        Static reference image. If 2D, align all images in `mov_stack` to same
        reference images. If 3D, align each image in `mov_stack` to
        corresponding index along first dimension of `ref`.
    mov_stack : np.ndarray
        3D image stack to align. Must have same shape as `ref` along the last
        two dimensions.
    static_mov : np.ndarray, optional
        2D static reference image. If provided, register `static_mov` to `ref`
        and apply transformation matrix to `mov_stack`.
        Defaults to None, in which case registration uses `mov_stack`.
    cycles : int, optional
        Number of times to repeat alignment.
        Defaults to 1.
    mode : str, optional
        Constraints of registration paradigm:
        - "translation" --> translation in X/Y directions.
        - "rigid" --> rigid transformations. Default option.
        - "rotation" --> rotation and dilation.
        - "affine" --> affine transformation.
        - "bilinear" --> bilinear transformation.
    order : int, optional
        Interpolation paradigm for transformation. Valid values are integers on
        range [0 (nearest-neighbor), 5 (bi-quintic)].
        Defaults to 0.
    cval : float, optional
        Fill value for warped pixels with no equivalent in `mov`.
        Defaults to 0.0.

    Returns
    -------
    mov_stack : np.ndarray
        Aligned stack.

    Examples
    --------
    """
    use_constant_matrix, matrix = False, None
    if static_mov is not None and ref.ndim == 2:
        use_constant_matrix = True
        _, matrix = align_image(
            ref, static_mov, cycles=cycles, mode=mode, order=order, cval=cval)

    for i in range(mov_stack.shape[0]):
        if use_constant_matrix:
            moved = transform(mov_stack[i], matrix, order=order, cval=cval)
        else:
            moved, _ = align_image(
                ref if ref.ndim == 2 else ref[i], mov_stack[i], cycles=cycles,
                mode=mode, order=order, cval=cval)

        mov_stack[i] = moved

    return mov_stack
