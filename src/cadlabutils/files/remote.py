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


def list_directory(
        dir_url: str,
        links_only: bool
):
    # list image paths on remote server
    soup = BeautifulSoup(requests.get(dir_url).text, 'html.parser')
    links = [l.get("href") for l in soup.find_all("a")]
    links = natsort.natsorted([
        dir_url + l for l in links if not
        any((l is None, l.endswith("../"), links_only and l.endswith("/")))])
    return links


def download_url(
        file_url: str,
        file_path: Path
):
    # access data via url download
    with requests.get(file_url, stream=True) as r:
        r.raise_for_status()

        # save data locally
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def read_tif_url(
        file_url: str
):
    # access data via url download
    response = requests.get(file_url)
    response.raise_for_status()

    # convert data directly to array
    array = tf.imread(BytesIO(response.content))
    return array


def read_text_url(
        file_url: str
):
    response = requests.get(file_url)
    response.raise_for_status()
    return response.text
