#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:06:21 2021
@author: ike
"""

import numpy as np
import skimage.util as sku
import skimage.transform as skt

from pystackreg import StackReg


# Instantiate registration machinery from pystackreg.
T_REG = {
    "translation": StackReg(StackReg.TRANSLATION),
    "rigid": StackReg(StackReg.RIGID_BODY),
    "rotation": StackReg(StackReg.SCALED_ROTATION),
    "affine": StackReg(StackReg.AFFINE),
    "bilinear": StackReg(StackReg.BILINEAR)
}


# Change dtype of transformed image
D_TYPE = {
    int: sku.img_as_int,
    float: sku.img_as_float64,
    np.uint8: sku.img_as_ubyte,
}


def _register(
        ref: np.ndarray,
        mov: np.ndarray,
        mode: str = "rigid"
):
    """Register an image to a static reference.

    Parameters
    ----------
    ref : np.ndarray
        2D static reference image.
    mov : np.ndarray):
        Image with same shape as `ref`. Registration maps mov onto ref.
    mode : str, optional
        Constraints of registration paradigm:
        - "translation" --> translation in X/Y directions.
        - "rigid" --> rigid transformations.
        - "rotation" --> rotation and dilation.
        - "affine" --> affine transformation.
        - "bilinear" --> bilinear transformation.
        Defaults to "rigid".

    Returns
    -------
    matrix : np.ndarray
        3x3 transformation matrix mapping `mov` onto `ref`.

    Examples
    --------
    >>> t_ref = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
    >>> t_mov = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 1]])
    >>> _register(t_ref, t_mov)
    array([[ 1., -0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    """
    matrix = T_REG[mode].register(ref, mov)
    return matrix


def _transform(
        mov: np.ndarray,
        matrix: np.ndarray,
        order: int = 0,
        dtype: type = np.uint8
):
    """Transform an image using a known transformation matrix.

    Parameters
    ----------
    mov : np.ndarray
        2D image to be transformed.
    matrix : np.ndarray
        3x3 transformation matrix with which to transform mov.
    order : int, optional
        Passed to constructor of skimage.transform.warp. Specifies
        interpolation paradigm for transformation. Valid values are integers on
        range [0 (nearest-neighbor), 5 (bi-quintic)].
        Defaults to 0.
    dtype : type, optional
        Datatype of warped image. Must be int, float, or np.uint8.
        Defaults to np.uint8.

    Returns
    -------
    moved : np.ndarray
        Transformed image.
    """
    moved = skt.warp(mov, matrix, order=order, mode="constant", cval=0)
    moved = D_TYPE[dtype](moved)
    return moved


def align_image(
        ref: np.ndarray,
        mov: np.ndarray,
        mode: str = "rigid",
        order: int = 0,
        conversion: str = "ubyte"
):
    """
    Align and transform an image to a static reference using constrained
    transformations.

    Note:
        align_image is a safe transformation function, meaning the
        transformation should occur as intended when accessing this function
        from a different script. _register and _transform should not be called
        from external scripts unless absolutely necessary as their behavior
        will be more constrained with nonstandard inputs.

    Args:
        ref (numpy.ndarray): 2D static reference image.
        mov (numpy.ndarray): Image to be aligned and transformed. Must have the
            same dimensions as the reference image.
        mode (str): Constraints of the registration paradigm. Passed to
            function call for _register. Valid inputs are "translation",
            "rigid", "rotation", "affine", "bilinear". See _register docstring
            for more information.
        order (int): Interpolation paradigm for transformation. Passed to
            function call for _transform. Valid inputs are 0-5. See _transform
            docstring for more information.
        conversion (str): Desired data type of transformed image. Passes to
            function call for _transform. Valid inputs are
            "int", "uint", "ubyte", "bool", "float", "float32", "float64". See
            _transform docstring for more information.

    Returns:
        numpy.ndarray: Transformed 2D image.
    """
    tmat = _register(ref, mov, mode)
    print(tmat)
    mov = _transform(mov, tmat, order, conversion)
    return mov


def align_hyperstack(
        ref_stack: np.ndarray,
        mov_stack: np.ndarray,
        channel: int = 0,
        mode: str = "rigid",
        order: int = 0,
        conversion: str = "ubyte"
):
    """
    Align and transform an image to a static reference using constrained
    transformations.

    Note:
        align_image is a safe transformation function, meaning the
        transformation should occur as intended when accessing this function
        from a different script. _register and _transform should not be called
        from external scripts unless absolutely necessary as their behavior
        will be more constrained with nonstandard inputs.

    Args:
        ref_stack (numpy.ndarray): 5D static reference hyperstack. Dimension
            conventions are as follows:
            - T --> May be of length 1 or of a length equivalent to that in the
                mov_stack. If 1, all images along the T dimension in mov_stack
                will be aligned to the same single reference in ref_stack.
            - Z --> May be of length 1 or of a length equivalent to that in the
                mov_stack. If 1, all images along the Z dimension in mov_stack
                will be aligned to the same single reference in ref_stack.
            - C --> Must be of length 1. Between channel alignment is not
                performed as channels are acquired simultaneously (ie. when
                imaging multiple fluorophores within a single cell), in which
                case all channels display the same distortions and require the
                same transformation to be aligned to a single reference.
            - Y --> Must be of a length equivalent to that in the mov_stack.
            - X --> Must be of a length equivalent to that in the mov_stack.
        mov_stack (numpy.ndarray): Image to be aligned and transformed.
        channel (int): Channel in mov_stack to use as a reference to derive
            transformation matrices in comparison to ref_stack. All other
            channels will be aligned in tandem. Defaults to 0.
            channel
        mode (str): Constraints of the registration paradigm. Passed to
            function call for _register. Valid inputs are "translation",
            "rigid", "rotation", "affine", "bilinear". See _register docstring
            for more information.
        order (int): Interpolation paradigm for transformation. Passed to
            function call for _transform. Valid inputs are 0-5. See _transform
            docstring for more information.
        conversion (str): Desired data type of transformed image. Passes to
            function call for _transform. Valid inputs are
            "int", "uint", "ubyte", "bool", "float", "float32", "float64". See
            _transform docstring for more information.

    Returns:
        numpy.ndarray: Transformed 2D image.

    Raises:
        NotImplementedError: ref_stack with multiple channels.
    """
    if ref_stack.shape[2] > 1:
        raise NotImplementedError(
            "Alignment with multi-channel reference not supported.")

    for t in range(mov_stack.shape[0]):
        for z in range(mov_stack.shape[1]):
            # if reference has a single timepoint, align to it
            t_index = min(t, ref_stack.shape[0] - 1)
            # if reference has single z value, align to it
            z_index = min(z, ref_stack.shape[1] - 1)
            ref = ref_stack[t_index, z_index, 0]
            # perform registration against reference using specified channel
            mov = mov_stack[t, z, channel]
            tmat = _register(ref, mov, mode=mode)
            # align all channels according to reference channel matrix
            for c in range(mov_stack.shape[2]):
                mov_stack[t, z, c] = _transform(mov, tmat, order, conversion)

    return mov_stack
