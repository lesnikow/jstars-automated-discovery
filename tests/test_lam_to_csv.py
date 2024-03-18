#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_lam_to_csv
Tests for lam_to_csv module.
"""

import csv
import os
import sys

sys.path.insert(0, "lam/")
from lam_to_csv import get_metadata_from_lroc_id, main


def test_get_metadata_from_lroc_id():
    """Test for our get_metadata_from_lroc_id method."""
    lroc_raw_id = "M104490494LC"
    metadata = get_metadata_from_lroc_id(lroc_raw_id)
    assert metadata["res"] == 1.486
    assert metadata["emission_angle"] == 11.6
    assert metadata["phase_angle"] == 67.77
    assert metadata["incidence_angle"] == 56.77

    assert metadata["ul_lat"] == 27.48
    assert metadata["ul_long"] == 3.38
    assert metadata["lr_lat"] == 24.96
    assert metadata["lr_long"] == 3.67

    lroc_raw_id = "M144524996LC"
    metadata = get_metadata_from_lroc_id(lroc_raw_id)
    assert metadata["emission_angle"] == 3.74

    lroc_raw_id = "M1167141636RC"
    metadata = get_metadata_from_lroc_id(lroc_raw_id)
    assert metadata["emission_angle"] == 1.19


def test_main(run_main_bool=True):
    """Test for our main method."""
    lam_dir = "data/processed/"
    out_csv_fp = "/tmp/test_lam_to_csv.csv"
    if run_main_bool:
        main(lam_dir=lam_dir, out_csv_fp=out_csv_fp)
    assert os.path.exists("/tmp/test_lam_to_csv.csv")

    # Test that we can open this out csv file, and read
    # that the expected columns are in the csv file fieldnames.
    with open(out_csv_fp, "r") as file:
        reader = csv.DictReader(file)
        expected_cols = [
            "lroc_id",
            "res",
            "altitude",
            "orbit_n",
            "nac_temp_telescope",
            "nac_temp_fpga",
            "slew_angle",
            "emission_angle",
            "phase_angle",
            "incidence_angle",
            "ul_lat",
            "ur_lat",
            "ll_lat",
            "lr_lat",
            "ul_long",
            "ur_long",
            "ll_long",
            "lr_long",
            "patch_id_from_parent_lam",
            "x_ul",
            "y_ul",
            "lat",
            "long",
            "anom_score",
        ]
        for col in expected_cols:
            assert col in reader.fieldnames

    # Assert that x_ul, y_ul values are ints, and that anom_score is a float,
    # for the first 128 rows, excluding the header row.
    # Also assert that the x_ul, y_ul, values are all non-negative.
    with open(out_csv_fp, "r") as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i == 128:
                break
            assert isinstance(int(row["x_ul"]), int)
            assert isinstance(int(row["y_ul"]), int)
            assert isinstance(float(row["anom_score"]), float)
            assert int(row["x_ul"]) >= 0
            assert int(row["y_ul"]) >= 0
            assert float(row["anom_score"]) >= 0
