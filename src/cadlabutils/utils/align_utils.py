#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:06:21 2021
Core code obtained from Vicky Mak, Barnhart Lab
Class structure and relative imports from Ike Ogbonna, Barnhart Lab
@author: ike
"""

import numpy as np
import skimage.util as sku
import skimage.transform as skt

from pystackreg import StackReg


"""
Helper functions for aligning stacks of imaging data using pystackreg backend,
a python implementation of the NIH Fiji plugin TurboReg. This code uses
pystackreg to register images (ie. determine the transofrmation matrix needed
to transform one image onto a dissimilar static reference image). Subsequent
transformations are performed using functions from the transfrom module of the
builtin skimage package.
"""


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
    "bool": sku.img_as_bool,
    "int": sku.img_as_int,
    "uint": sku.img_as_uint,
    "ubyte": sku.img_as_ubyte,
    "float": sku.img_as_float,
    "float32": sku.img_as_float32,
    "float64": sku.img_as_float64
}


def _register(
        ref: np.ndarray,
        mov: np.ndarray,
        mode: str = "rigid"
):
    """
    Register an image to a static reference using constrained transformations.

    Args:
        ref (numpy.ndarray): 2D static reference image.
        mov (numpy.ndarray): Image with same dimesnions as ref. Registration
            maps mov onto ref.
        mode (str): Constraints of registration paradigm:
            - "translation" --> translation in X/Y directions.
            - "rigid" --> rigid transformations.
            - "rotation" --> rotation and dilation.
            - "affine" --> affine transformation.
            - "bilinear" --> bilinear transformation.

    Returns:
        numpy.ndarray: 2D transformation matrix mapping mov onto ref.

    Raises:
        ValueError: ref array is not 2D.
        ValueError: mov array has different dimensions than ref.

    Example:
        >>> test_ref = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
        >>> test_mov = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 1]])
        >>> transformation_matrix = _register(test_ref, test_mov)
        >>> transformation_matrix
        array([[ 1., -0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])

    Error Examples:
        # ref is not 2D
        >>> invalid_ref = np.array([[[1, 0, 0], [1, 1, 0], [0, 1, 0]]])
        >>> test_mov = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 1]])
        >>> transformation_matrix = _register(invalid_ref, test_mov)
        Traceback (most recent call last):
            ...
        ValueError: Reference image must be 2D array.

        # mov has a different shape than ref
        >>> test_ref = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
        >>> invalid_mov = np.zeros((5,6))
        >>> transformation_matrix = _register(test_ref, invalid_mov)
        Traceback (most recent call last):
            ...
        ValueError: Move image must have same shape as reference.
    """
    if ref.ndim != 2:
        raise ValueError("Reference image must be 2D array.")

    if ref.shape != mov.shape:
        raise ValueError("Move image must have same shape as reference.")

    tmat = T_REG[mode].register(ref, mov)
    return tmat


def _transform(
        mov: np.ndarray,
        tmat: np.ndarray,
        order: int = 0,
        conversion: str = "ubyte"
):
    """
    Transform an image using a known transformation matrix.

    Args:
        mov (numpy.ndarray): 2D image to be transformed.
        tmat (numpy.ndarray): 2D transformation matrix with which to transform
            mov. Matrix should be a 3x3.
        order (int): Passes to constructor of skimage.transform.warp. Specifies
            interpolation paradigm for transformation. Accepted values include:
            - 0 --> Nearest-neighbor (default)
            - 1 --> Bi-linear
            - 2 --> Bi-quadratic
            - 3 --> Bi-cubic
            - 4 --> Bi-quartic
            - 5 --> Bi-quintic

            Of note, nearest-neightbor interpolation is preferred for low-bit
            ("pixelated") images as it preserves the original pixel values.
        conversion (str): function used to alter the data type of
            the transformed image. supported inputs are:
            - bool --> boolean
            - int --> 16-bit signed integer
            - uint --> 16-bit unsigned integer
            - ubyte --> 8-bit unsigned integer (default)
            - float --> floating point
            - float32 --> single-precision (32-bit) floating point
            - float64 --> double-precision (64-bit) floating point

    Returns:
        numpy.ndarray: Transformed 2D image.
    """
    # transform image with nearest neighbor interpolation using skimage
    mov = skt.warp(mov, tmat, order=order, mode="constant", cval=0)
    # convert foating-point image to an unsigned 8-bit image
    mov = D_TYPE[conversion](mov)
    return mov


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
