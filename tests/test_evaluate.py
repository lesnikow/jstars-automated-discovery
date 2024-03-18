#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_evalute 
Tests for the evaluate module.
"""

import os
import pytest
import sys

sys.path.insert(0, "lam/")
from evaluate import get_label_lines_lroc_raw_ids_lists, write_out_lroc_raw_urls
from evaluate import download_lroc_raws, make_top_anom_patches_grid


def test_get_label_lines_lroc_raw_ids_lists():
    """Test get_label_lines_lroc_raw_ids_lists."""
    mission_str = "ap17"
    eval_labels_home = "data/interim/metadata/"
    labels_fp = os.path.join(eval_labels_home, mission_str + "_labels.csv")

    lroc_raw_ids_list, label_lines = get_label_lines_lroc_raw_ids_lists(labels_fp)

    # Assert expected lengths.
    assert len(lroc_raw_ids_list) >= 2
    assert len(label_lines) >= 2
    assert len(lroc_raw_ids_list) == len(label_lines)

    # Assert that the first line of lroc_raw_ids_list
    # is a string that begings with 'M' and ends with 'C'.
    assert lroc_raw_ids_list[0].startswith("M")
    assert lroc_raw_ids_list[0].endswith("C")

    # Assert that the first line of label_lines
    # is a list that has 3 elements, the first of which
    # is a string that begins with 'M' and ends with 'C',
    # and the last two of which are positive integers.
    assert len(label_lines[0]) == 3
    assert label_lines[0][0].startswith("M")
    assert label_lines[0][0].endswith("C")
    assert isinstance(label_lines[0][1], int)
    assert isinstance(label_lines[0][2], int)
    assert label_lines[0][1] >= 0
    assert label_lines[0][2] >= 0


def test_write_out_lroc_raw_urls():
    """Test write_out_lroc_raw_urls."""
    mission_str = "ap17"
    eval_labels_home = "data/interim/metadata/"
    labels_fp = os.path.join(eval_labels_home, mission_str + "_labels.csv")
    lroc_raw_ids_list, label_lines = get_label_lines_lroc_raw_ids_lists(labels_fp)
    out_lroc_raws_urls_fp = write_out_lroc_raw_urls(mission_str, lroc_raw_ids_list)
    assert os.path.exists(out_lroc_raws_urls_fp)
    assert os.path.getsize(out_lroc_raws_urls_fp) > 0

    # Assert that the file at out_lroc_raws_urls_fp
    # has the same number of lines as lroc_raw_ids_list
    # and that the first line contains a web url.
    with open(out_lroc_raws_urls_fp, "r") as f:
        lines = f.readlines()
        assert len(lines) == len(lroc_raw_ids_list)
        assert lines[0].startswith("http")


@pytest.mark.skip(reason="This download test takes a long time to run.")
def test_download_lroc_raws():
    """Test download_lroc_raws."""
    mission_str = "ap17"
    eval_labels_home = "data/interim/metadata/"
    labels_fp = os.path.join(eval_labels_home, mission_str + "_labels.csv")
    lroc_raw_ids_list, label_lines = get_label_lines_lroc_raw_ids_lists(labels_fp)
    out_lroc_raws_urls_fp = write_out_lroc_raw_urls(mission_str, lroc_raw_ids_list)

    download_raws_bool = True
    lroc_raws_save_fp = download_lroc_raws(
        mission_str, out_lroc_raws_urls_fp, download_raws_bool
    )

    # Assert that the directory at lroc_raws_save_fp
    # has the same number of files as lroc_raw_ids_list.
    assert os.path.exists(lroc_raws_save_fp)
    assert len(os.listdir(lroc_raws_save_fp)) == len(lroc_raw_ids_list)


def test_inference_on_lroc_raws():
    """Test inference_on_lroc_raws."""
    pass


def test_make_top_anom_patches_grid():
    """Test make_top_anom_patches_grid."""
    pass
