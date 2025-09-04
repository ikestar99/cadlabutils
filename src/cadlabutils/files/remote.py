#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:00:00 2025
@author: ike
"""


import natsort
import requests
import tifffile as tf

from io import BytesIO
from bs4 import BeautifulSoup
from pathlib import Path


def _request_response(
        url: str
):
    response = requests.get(url)
    response.raise_for_status()
    return response


def list_directory(
        url: str,
        links_only: bool
):
    """List files from a remote directory URL with natural sorting.

    Parameters
    ----------
    url : str
        URL of the remote directory. Must return an HTML page with links.
    links_only : bool
        If True, exclude directories (links ending with '/') from the listing.

    Returns
    -------
    links : list[str]
        A naturally sorted list of file URLs from `dir_url`.

    Notes
    -----
    - Excludes parent directory links ('../').
    - Uses BeautifulSoup for HTML parsing and natsort for natural sorting.
    """
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    links = [n.get("href") for n in soup.find_all("a")]
    links = natsort.natsorted([
        url + n for n in links if not
        any((n is None, n.endswith("../"), links_only and n.endswith("/")))])
    return links


def download_url(
        url: str,
        file: Path
):
    """Download a file from a URL and save it locally.

    Parameters
    ----------
    url : str
        URL of the file to download.
    file : Path
        Local file path where the downloaded file will be saved.
    """
    with requests.get(url, stream=True) as r, open(file, 'wb') as f:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def read_tif_url(
        url: str
):
    """Read a TIFF image from a remote URL into an array.

    Parameters
    ----------
    url : str
        URL of image file (.tif).

    Returns
    -------
    arr : numpy.ndarray
        Image data as an array.
    """
    # convert data directly to array
    arr = tf.imread(BytesIO(_request_response(url).content))
    return arr


def read_text_url(
        url: str
):
    """Read text contents of a remote file via URL.

    Parameters
    ----------
    url : str
        URL of text-like file.

    Returns
    -------
    str
        Contents of the text file as a string.

    Notes
    -----
    - Works with files whose content can be represented as a string, as with
      .csv, .txt, and even .swc files.
    """
    text = _request_response(url).text
    return text
