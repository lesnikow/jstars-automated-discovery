#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""lam_to_csv module
Defines our lam_to_csv module for outputting
csv files to be used in our GIS software,
from the lam files outputted by our inference module.
"""

import logging
import os
import sys
import time
import bs4
import requests


def get_metadata_from_lroc_id(lroc_raw_id):
    """Gets the metadata from the lroc_raw_id."""
    search_url_base = "https://wms.lroc.asu.edu/lroc/view_lroc/LRO-L-LROC-3-CDR-V1.0/"
    search_url = search_url_base + lroc_raw_id

    for i in range(32):
        try:
            search_page = requests.get(search_url)
            logging.info(search_page.status_code)
            if search_page.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            logging.info("We got a requests.exceptions.ConnectionError. Trying again.")
            time.sleep(2)
    html = search_page.text
    soup = bs4.BeautifulSoup(html, "html.parser")

    metadata = {}
    metadata["res"] = float(soup.find(string="Resolution").find_next_sibling().text)
    metadata["res"] = round(metadata["res"], 3)
    metadata["alt"] = float(
        soup.find(string="Spacecraft altitude").find_next_sibling().text
    )
    metadata["orb_n"] = int(soup.find(string="Orbit number").find_next_sibling().text)
    metadata["nac_temp_tele"] = float(
        soup.find(string="Nac temperature telescope").find_next_sibling().text
    )
    metadata["nac_temp_fpga"] = float(
        soup.find(string="Nac temperature fpga").find_next_sibling().text
    )
    metadata["slew_angle"] = round(
        float(soup.find(string="Slew angle").find_next_sibling().text), 2
    )
    metadata["emission_angle"] = float(
        soup.find(string="Emission angle").find_next_sibling().text
    )
    metadata["phase_angle"] = float(
        soup.find(string="Phase angle").find_next_sibling().text
    )
    metadata["incidence_angle"] = float(
        soup.find(string="Incidence angle").find_next_sibling().text
    )
    metadata["ul_lat"] = float(
        soup.find(string="Upper left latitude").find_next_sibling().text
    )
    metadata["ur_lat"] = float(
        soup.find(string="Upper right latitude").find_next_sibling().text
    )
    metadata["ll_lat"] = float(
        soup.find(string="Lower left latitude").find_next_sibling().text
    )
    metadata["lr_lat"] = float(
        soup.find(string="Lower right latitude").find_next_sibling().text
    )
    metadata["ul_long"] = float(
        soup.find(string="Upper left longitude").find_next_sibling().text
    )
    metadata["ur_long"] = float(
        soup.find(string="Upper right longitude").find_next_sibling().text
    )
    metadata["ll_long"] = float(
        soup.find(string="Lower left longitude").find_next_sibling().text
    )
    metadata["lr_long"] = float(
        soup.find(string="Lower right longitude").find_next_sibling().text
    )
    return metadata


def write_output_csv(lam_fp, out_csv_fp):
    """Writes the output csv file from the lam file.
    Has the following columns: lroc_id, resolution, altitude, orbit number,
    nac temperature telescope, nac temperature fpga, slew angle, emission angle,
    phase angle, incidence angle, upper left latitude, upper right latitude,
    lower left latitude, lower right latitude, upper left longitude,
    upper right longitude, lower left longitude, lower right longitude,
    x, y, anom_score.
    """
    with open(lam_fp, "r") as lam_file:
        lines = lam_file.readlines()
        header_length = 9
        header, rest = lines[:header_length], lines[header_length:]
        lroc_id = str(header[0].strip())
        total_y, total_x = int(header[1]), int(header[2])
        total_y_inferenced = int(header[3].strip())
        total_x_inferenced = int(header[4].strip())
        total_y_patches = int(float(header[5].strip()))
        total_x_patches = int(float(header[6].strip()))
        total_patches = int(float(header[7].strip()))
        logging.info("lroc_id: %s", lroc_id)
        logging.info("total_y: %s", total_y)
        logging.info("total_x: %s", total_x)
        logging.info("total_y_inferenced: %s", total_y_inferenced)
        logging.info("total_x_inferenced: %s", total_x_inferenced)
        logging.info("total_y_patches: %s", total_y_patches)
        logging.info("total_x_patches: %s", total_x_patches)
        logging.info("total_patches: %s", total_patches)

        metadata = get_metadata_from_lroc_id(lroc_id)

        with open(out_csv_fp, "a+") as out_file:
            # Approach 2 for start and end bounds: Take averages.
            # Assumption: The inferenced lroc raw parent images
            # are aligned along the lunar coordinate grid.
            # This assumption starts to break down when we
            # consider later lro science mission extensions.
            long, lat = metadata["ul_long"], metadata["ul_lat"]
            long_start = (metadata["ul_long"] + metadata["ll_long"]) / 2
            long_end = (metadata["ur_long"] + metadata["lr_long"]) / 2
            lat_start = (metadata["ul_lat"] + metadata["ur_lat"]) / 2
            lat_end = (metadata["ll_lat"] + metadata["lr_lat"]) / 2

            long_stride = (long_end - long_start) / total_x_patches
            lat_stride = (lat_end - lat_start) / total_y_patches

            logging.info("long_start: %s", long_start)
            logging.info("long_end: %s", long_end)
            logging.info("long_stride: %s", long_stride)
            logging.info("lat_start: %s", lat_start)
            logging.info("lat_end: %s", lat_end)
            logging.info("lat_stride: %s", lat_stride)

            # Catch the case when long_end minus long_start is larger in magnitude
            # than our expected max value.
            # This is due to the fact that we could go from 360 to 0 degrees
            # or vice versa.
            max_long_diff_expected = 1.0
            if abs(long_end - long_start) > max_long_diff_expected:
                raise ValueError(
                    "long_end - long_start is larger than expected. "
                    "long_end: %s, long_start: %s, max_long_diff_expected: %s",
                    long_end,
                    long_start,
                    max_long_diff_expected,
                )

            patch_id_from_parent_lam = 0
            x_ul, y_ul = 0, 0
            x_stride, y_stride = 64, 64
            for line in rest:
                anom_score = line.strip()
                out_vals = [
                    lroc_id,
                    metadata["res"],
                    metadata["alt"],
                    metadata["orb_n"],
                    metadata["nac_temp_tele"],
                    metadata["nac_temp_fpga"],
                    metadata["slew_angle"],
                    metadata["emission_angle"],
                    metadata["phase_angle"],
                    metadata["incidence_angle"],
                    metadata["ul_lat"],
                    metadata["ur_lat"],
                    metadata["ll_lat"],
                    metadata["lr_lat"],
                    metadata["ul_long"],
                    metadata["ur_long"],
                    metadata["ll_long"],
                    metadata["lr_long"],
                    patch_id_from_parent_lam,
                    y_ul,
                    x_ul,
                    lat,
                    long,
                    anom_score,
                ]
                out_line = ",".join([str(x) for x in out_vals])
                out_file.write(out_line + "\n")

                x_ul += x_stride
                long += long_stride
                if x_ul >= total_x_inferenced:
                    x_ul = 0
                    long = long_start
                    y_ul += y_stride
                    lat += lat_stride

                patch_id_from_parent_lam += 1


def main(feature_str=None, lam_dir=None, out_csv_fp=None):
    """The main method.
    Iterate through the lam files in the lam directory,
    calling write_output_csv on each one.
    """
    if feature_str is None:
        feature_str = "ap17"
    if lam_dir is None:
        lam_dir = "data/processed/"
    if out_csv_fp is None:
        out_csv_fp = "data/processed/lam_to_csvs/lam_to_csv_{}.csv".format(feature_str)

    log_filepath = os.path.join("logs", f"lam_to_csv_{time.time()}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting lam_to_csv...")
    logging.info("feature_str: %s", feature_str)
    logging.info("lam_dir: %s", lam_dir)
    logging.info("out_csv_fp: %s", out_csv_fp)

    if os.path.exists(out_csv_fp):
        os.remove(out_csv_fp)

    with open(out_csv_fp, "a+") as out_file:
        col_headers = [
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
            "y_ul",
            "x_ul",
            "lat",
            "long",
            "anom_score",
        ]
        for col_header in col_headers:
            logging.info("col_header: %s", col_header)
        out_file.write(",".join(col_headers) + "\n")

    count_lam = 0
    # Iterate through the lam files in the lam directory
    lam_dir = os.path.join(lam_dir, feature_str)
    for root, dirs, files in os.walk(lam_dir):
        for file in files:
            if file.endswith(".lam"):
                count_lam += 1
                logging.info("Processing lam file %s", file)
                lam_file = os.path.join(root, file)
                # Call write_output_csv on each lam file
                write_output_csv(lam_file, out_csv_fp)
                logging.info("Finished processing lam file %s", file)
                logging.info("count_lam: %s", count_lam)

    # log the header and the first two lines of the output csv
    with open(out_csv_fp, "r") as out_file:
        lines = out_file.readlines()
        logging.info("header: %s", lines[0])
        logging.info("first line: %s", lines[1])
        logging.info("second line: %s", lines[2])

    logging.info("Wrote output csv to %s", out_csv_fp)
    logging.info("Finished lam_to_csv.")
    return out_csv_fp


if __name__ == "__main__":
    main()
