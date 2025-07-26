#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 03:04:49 2021

@author: ike
"""


import os
import shutil
import os.path as op

from glob import glob as gg


"""
Helper functions for manipulating file paths and directories.
"""


def get_path(
        *args: str,
        ext: str = ""
):
    """
    Convert arbitrary number of filenames or subpaths into one filepath.
    
    Note:
        get_path was designed to normalize file paths in a platform agnostic
        manner using existinging methods in the builtin os module.

    Args:
        *args (str): Filenames or filepaths to be combined, in order.
        ext (str, optional): File extension. Defaults to "".

    Returns:
        str: Path to the desired file or directory.

    Examples:
        # create path to subdirectory
        >>> test_path = get_path("dir1", "file1")
        >>> print(test_path)
        dir1/file1

        # create path to file
        >>> test_path = get_path("dir1/dir2/", "dir3\\dir4", "dir5", ext="txt")
        >>> print(test_path)
        dir1/dir2/dir3/dir4/dir5.txt
    """
    # normalize file separators to default for operating system
    path = [arg.replace("\\", op.sep).replace("/", op.sep) for arg in args]
    # join arguments with file separator and prune redundant separators
    path = op.normpath(op.sep.join(path))
    # add file extension if applicable
    path = (".".join([path, ext]) if ext != "" else path)
    return path


def get_parent(
        path: str,
        num: int = 1
):
    """
    Find parent directory of a file or subdirectory.

    Args:
        path (str): Filepath or subdirectory from which to find a parent.
        num (int, optional): How many subdirectories to traverse before
            returning parent. Defaults to 1.

    Returns:
        str: Path to the parent directory.

    Raises:
        TypeError: If input is not a path string.
        ValueError: If num input is less than 1.
        ValueError: If path has fewer than num directories.

    Examples:
        >>> test_path = "/dir1/dir2/file.txt"
        >>> test_parent = get_parent(test_path)
        >>> print(test_parent)
        /dir1/dir2

        # Recursion to find higher level directories
        >>> test_parent = get_parent(test_path, num=2)
        >>> print(test_parent)
        /dir1

    Error Examples:
        # path is not a viable directory or file path
        >>> invalid_path = "foo.bar"
        >>> get_parent(invalid_path)
        Traceback (most recent call last):
            ...
        TypeError: Input string must be a path.

        # num input is less than 1
        >>> get_parent(test_path, num=0)
        Traceback (most recent call last):
            ...
        ValueError: num input must be greater than or equal to 1.

        # num larger than number of subdirectories in path
        >>> get_parent(test_path, num=10)
        Traceback (most recent call last):
            ...
        ValueError: Too many recursions for input path.
    """
    if op.sep not in path:
        raise TypeError("Input string must be a path.")

    if num < 1:
        raise ValueError("num input must be greater than or equal to 1.")

    if path.count(op.sep) < num:
        raise ValueError("Too many recursions for input path.")

    path = op.dirname(path)
    path = (path if num <= 1 else get_parent(path, num - 1))
    return path


def get_name(
        path: str
):
    """
    Find name of a file or subdirectory.

    Args:
        path (str): Filepath or subdirectory from which to find a name.

    Returns:
        str: Name of the given file or subdirectory.

    Examples:
        >>> test_name = get_name("/dir1/dir2/file.txt")
        >>> print(test_name)
        file.txt
    """
    name = op.basename(path)
    return name


def change_extension(
        path: str,
        ext: str = ""
):
    """
    Add or change the extension of a filepath. Can be used to strip extension
    off of a path as well.

    Args:
        path (str): Filepath to modify.
        ext (str, optional): Extension to add to the filepath. Defaults to "".

    Returns:
        str: Filepath with the old extension removed and any new extension
            added.

    Examples:
        # remove extension
        >>> test_path = change_extension("file.txt")
        >>> print(test_path)
        file

        # add new extension
        >>> test_path = change_extension("file.txt", ext="jpg")
        >>> print(test_path)
        file.jpg
    """
    # Remove current file extension if present
    path = (path[:path.rindex(".")] if "." in path else path)
    # Add new file extension if present
    path = (".".join([path, ext]) if ext != "" else path)
    return path


def check_path(
        path: str
):
    """
    Check if a given path exists, whether it's a file or a folder.

    Args:
        path (str): Path to check for existence.

    Returns:
        bool: True if the path exists, False otherwise.

    Examples:
        >>> invalid_path = "/This/folder/does/not/exist"
        >>> invalid_path = check_path_exists(invalid_path)
        >>> print(invalid_path)
        False

        >>> test_path = "/This/file/does/exist.pdf"
        >>> test_path = check_path_exists(test_path)
        >>> print(test_path)
        True
    """
    return op.exists(path)


def make_directory(
        path: str
):
    """
    If a necessary directory does not exist, create it.

    Args:
        path (str): Path to the necessary directory.

    Raises:
        ValueError: If path represents a file rather than a directory.

    Example:
        test_path = ".../dir1/dir2
        make_directory(test_path)

    Error Example:
        >>> invalid_path = "/dir1/dir2/foo.bar"
        >>> make_directory(invalid_path)
        Traceback (most recent call last):
            ...
        ValueError: Path must be a valid directory.
    """
    if "." in path:
        raise ValueError("Path must be a valid directory.")

    if not op.isdir(path):
        os.makedirs(path, exist_ok=False)


def make_parent(
        path: str
):
    """
    If a necessary parent directory does not exist, create it.

    Args:
        path (str): Filepath for which to check parent.

    Example:
        test_path = "/dir1/dir2/file.txt"
        make_parent(test_path)
    """
    make_directory(get_parent(path))


def move_path(
        source: str,
        destination: str
):
    """
    Move a filepath or subdirectory to a new location. Intended behavior
    includes a native check that the directory tree specified by  destination
    already exists. Should this not be the case, the directory tree will be
    created prior to performing the move.

    Args:
        source (str): Path to the file or subdirectory to be moved.
        destination (str): Desired path to the file or subdirectory.

    Raises:
        FileExistsError: If destination already exists.
        FileNotFoundError:

    Example:
        test_source = "/dir1/file.txt"
        test_destination = "/dir2/file.txt"
        move_path(test_source, test_destination)

    Error Examples:
        # source directory does not exist
        invalid_source = "fake/path/doesn't/exist"
        move_path(invalid_source, test_destination)
        Traceback (most recent call last):
            ...
        FileNotFoundError: Source file does not exist.

        # destination directory already exists
        make_directory(test_destination)
        move_path(test_source, test_destination)
        Traceback (most recent call last):
            ...
        FileExistsError: Destination file already exists.
    """
    check = (op.isfile(source) or op.isdir(source))
    if not check:
        raise FileNotFoundError("Source file does not exist.")

    check = (op.isfile(destination) or op.isdir(destination))
    if check:
        raise FileExistsError("Destination file already exists.")

    make_parent(destination)
    shutil.move(src=source, dst=destination)


def get_glob(
        *args: str,
        ext: str = ""
):
    """
    glob function from glob package with the following additions:
    - Return any matches in alphabetical order.
    - Return None if no matches found.

    Note:
        glob.glob backend performs search and pattern matching through relevant
        directories. Any wildcards (ie. "*" for unknown file/directory names)
        should be included in args for relevant functionality in glob.glob
        function call. including "**" in any input argument will automatically
        initiate a recursive search.

    Args:
        *args (str): Passed to get_path function call.
        ext (str, optional): Passed to get_path function call. Defaults to "".

    Returns:
        list: List of paths that match the desired pattern.
        None: None if list is empty.

    Examples:
        >>> get_glob("dir1", "file1")
        ['dir1/file1']

        # Unknown name wildcard "*"
        >>> get_glob("dir1", "dir2", "*", ext="txt")
        ['dir1/dir2/foo.txt', 'dir1/dir2/bar.txt']

        # Unknown subdirectories wildcard "**"
        >>> get_glob("dir1", "dir2", "**", "*", ext="txt")
        ['/dir1/dir2/file.txt', '/dir1/dir2/subdir/file.txt']
    """
    path = get_path(*args, ext=ext)
    path_list = sorted(gg(path, recursive=("**" in path)))
    path_list = (path_list if len(path_list) > 0 else None)
    return path_list


def get_folders(
        paths: list
):
    """
    Retrieve folders from a list of paths.

    Args:
        paths (list): List of file paths.

    Returns:
        list: List of folders from the input files.

    Raises:
        TypeError: If any element in the list is not a string.

    Example:
        get_folders(['/dir1/file1.txt', '/dir1/dir2'])
        ['/dir1/dir2']

    Error Example:
        >>> get_folders(["/dir1/dir2", 5])
        Traceback (most recent call last):
            ...
        TypeError: All elements of list must be strings.
    """
    check = [isinstance(p, str) for p in paths]
    if not all(check):
        raise TypeError("All elements of list must be strings.")

    paths = [p for p in paths if op.isdir(p)]
    return paths


# TODO: Create functions to delete files and directories on host computer
# def remove_parent(
#         path: str
# ):
#     """
#     If the parent directory of the filepath is empty, delete it.
#
#     Args:
#         path (str): Filepath for which to check parent.
#
#     Examples:
#         >>> remove_parent("/dir1/dir2/file.txt")
#     """
#     parent = get_parent(path)
#     if op.isdir(parent) and not os.listdir(parent):
#         shutil.rmtree(parent)
