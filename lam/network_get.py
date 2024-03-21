#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""network_get module
Defines our network_get module for downloading LROC raw images
from the ASU PDS server.
"""

import logging
import os
import random
import sys
import time
import urllib3.exceptions
import requests


def main(img_urls_list_fp=None, out_home_fp=None):
    """The main method of this module."""
    LOG_FILENAME = f"logs/network_get_log_{int(time.time())}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_FILENAME), logging.StreamHandler(sys.stdout)],
    )
    if not img_urls_list_fp:
        img_urls_list_fp = "/home/adam/out_shard.lst"
    if not out_home_fp:
        out_home_fp = os.path.join("/nvme", "raws", "class_0")

    img_urls_list_file_object = open(img_urls_list_fp)
    img_urls_list = img_urls_list_file_object.readlines()
    random.shuffle(img_urls_list)

    requests.adapters.DEFAULT_RETRIES = 2
    read_timeout = 40

    img_count = 0
    logging.debug(f"img_urls_list:{img_urls_list} ")
    for img_url in img_urls_list:
        img_count += 1
        try:
            lroc_id = img_url.split("/")[-1]
            out_path = os.path.join(out_home_fp, lroc_id)
            # Hack to get rid of newline at end of out_path to get correct
            # out file names in our Python out file object write calls.
            out_path = out_path[:-1]
            logging.info(f"Image count: {img_count} of {len(img_urls_list)}")
            logging.info(f"Starting to download LROC raw at: {img_url.strip()} ")
            logging.info(f"out_path: {out_path}")

            response = requests.get(img_url[:-1], stream=True, timeout=read_timeout)
            logging.info(f"Response status code: {response.status_code}")
            if response.status_code != 200:
                raise Exception("ERROR: %s does not exist!" % (img_url))
            with open(out_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                logging.info(f"Downloaded LROC raw to {out_path}.")
        except (
            ConnectionResetError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            urllib3.exceptions.IncompleteRead,
            urllib3.exceptions.ProtocolError,
            urllib3.exceptions.ReadTimeoutError,
        ) as e:
            logging.info(f"Error: {e}")


if __name__ == "__main__":
    main()
