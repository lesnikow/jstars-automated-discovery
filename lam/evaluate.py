#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""evaluate module
Defines the evaluate module for evaluating trained models. This module is used
to generate metrics, plots, and statistics on test sets, e.g. precision-recall
curves on collections of Apollo landing sites or rockfalls. This module is used
to produce many of the figures in our publication using various options. This
module splits this evaluate functionality from our main module. This module
supports a wide range of operations including downloading raw images, running
model inference, making anomaly scores, and generating various statistics and
tests to evaluate model performance.

Features:
- Downloading of raw images based by providing feature string.
- Inference on downloaded images using a specified trained model.
- Generation of anomaly scores for evaluation.
- Creation of precision-recall curves to assess model accuracy.
- Generation of KDE plots for anomaly score distribution visualization.
- Execution of KS tests for statistical comparison of distributions.
- Creation of t-SNE plots for visualizing high-dimensional data.
- Support for making grid maps and top anomalous patch grids.

Examples:

To download raw images with default settings on the 'ap17' feature, run:
python evaluate.py --feature_str ap17 --download_raws
To inference a model with default settings on the 'ap17' feature, and generate
anomlay scores, run:
python evaluate.py --feature_str ap17 --inference
To generate a precision-recall curve and a KDE plot for the 'crater' feature:
python evaluate.py --feature_str crater --make_pr_curve --make_kde_plot
To perform a KS test on anomaly scores for the 'rockfall' feature:
python evaluate.py --feature_str rockfall --make_ks_test
"""

import argparse
import csv
import datetime
import logging
import math
import os
import random
import shutil
import sys
import time
from typing import List, Tuple

import bs4
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats
import seaborn as sns
import tqdm

from inference import main as inference
from lam_to_csv import main as lam_to_csv
from metrics import PrecisionRecallCurve
from network_get import main as network_get

Image.MAX_IMAGE_PIXELS = None


def get_label_lines_lroc_raw_ids_lists(eval_labels_fp):
    """Get label_lines and lroc_raw_ids_lists from eval_labels_fp."""
    lroc_raw_ids_list, label_lines = [], []
    with open(eval_labels_fp, "r", encoding="utf-8") as label_file:
        n = 0
        for line in label_file:
            n += 1
            if n == 1:
                continue
            lroc_raw_id, x, y = line.split(",")
            x, y = int(float(x)), int(float(y))
            if x < 64 or y < 64 or x >= 4992:
                # Continue past images where the landing site is too close to the
                # left edge or top edge or right edge of the image.
                continue
            label_lines.append([lroc_raw_id, x, y])
            lroc_raw_ids_list.append(lroc_raw_id)
            logging.debug((lroc_raw_id, x, y))
    logging.debug("label_lines is %s", label_lines)
    logging.debug("lroc_raw_ids_list is %s", lroc_raw_ids_list)
    return lroc_raw_ids_list, label_lines


def write_out_lroc_raw_urls(feature_str, lroc_raw_ids_list):
    """Write out lroc_raw_urls file from lroc_raw_ids_list, calling the PDS
    to translate from lroc raw ids to lroc raw urls.
    """
    out_lroc_raws_urls_fp = f"/tmp/lroc_raws_feature_{feature_str}_urls.lst"
    if os.path.exists(out_lroc_raws_urls_fp):
        os.remove(out_lroc_raws_urls_fp)
    search_url_base = "https://wms.lroc.asu.edu/lroc/view_lroc/LRO-L-LROC-3-CDR-V1.0/"
    with open(out_lroc_raws_urls_fp, "a", encoding="utf-8") as out_file:
        for lroc_raw_id in lroc_raw_ids_list:
            search_url = search_url_base + lroc_raw_id
            logging.debug(search_url)
            for i in range(32):
                try:
                    search_page = requests.get(search_url)
                    logging.info(
                        "search_page status code is %s", search_page.status_code
                    )
                    if search_page.status_code == 200:
                        break
                except requests.exceptions.ConnectionError:
                    logging.info(
                        "We got a requests.exceptions.ConnectionError. Trying again."
                    )
                    time.sleep(2)
            html = search_page.text
            soup = bs4.BeautifulSoup(html, "html.parser")
            for link in soup.findAll("a"):
                link_href = link.get("href")
                if link_href[-8:] == "_pyr.tif":
                    logging.debug(link_href)
                    break
            out_file.write("http:" + link_href + os.linesep)
    return out_lroc_raws_urls_fp


def download_lroc_raws(feature_str, out_lroc_raws_urls_fp, download_raws_bool):
    """Downloads lroc_raws from feature_str and out_lroc_raws_urls_fp, calling
    network_get to download the lroc_raws."""
    lroc_raws_save_fp = os.path.join("data", "raw", feature_str, "class_0")
    if download_raws_bool:
        if os.path.exists(lroc_raws_save_fp):
            shutil.rmtree(lroc_raws_save_fp)
        if not os.path.exists(lroc_raws_save_fp):
            os.makedirs(lroc_raws_save_fp)
        network_get(
            img_urls_list_fp=out_lroc_raws_urls_fp, out_home_fp=lroc_raws_save_fp
        )
    else:
        logging.info(
            "Skipping downloading raws, since download_raws_bool is %s",
            download_raws_bool,
        )
    return lroc_raws_save_fp


def inference_on_lroc_raws(feature_str, lroc_raws_save_fp, model_fp, inference_bool):
    """Inference on lroc_raws from feature_str, lroc_raws_save_fp, model_fp."""
    out_inference_fp = os.path.join("data", "processed", feature_str)
    if inference_bool:
        lroc_raws_save_fp = "/".join(lroc_raws_save_fp.split("/")[:-1])
        logging.debug(f"lroc_raws_save_fp: {lroc_raws_save_fp}")
        try:
            inference(
                raw_imgs_fp=lroc_raws_save_fp,
                model_fp=model_fp,
                out_fp_base=out_inference_fp,
            )
        except RuntimeError as e:
            logging.info(
                "We got an inference module RuntimeError. Passing without further error handling."
            )
            logging.info(e)
    else:
        logging.info("Skipping inference, since inference_bool is %s", inference_bool)
    return out_inference_fp


def generate_anom_scores_and_labels(label_lines, out_inference_fp):
    """Generate anom_scores, anom_scores_positives, and labels from
    label_lines and out_inference_fp."""
    anom_scores, labels, anom_scores_positives, anom_scores_positives_list = (
        [],
        [],
        {},
        [],
    )
    for label_line in label_lines:
        logging.debug("label_line is %s", label_line)
        lroc_id, label_x_pixel, label_y_pixel = label_line
        with open(
            os.path.join(out_inference_fp, lroc_id + ".lam"), "r", encoding="utf-8"
        ) as lam_file:
            head = [next(lam_file).strip() for _ in range(9)]
            logging.debug("head is %s", head)
            lam_size_y, lam_size_x = int(float(head[5])), int(float(head[6]))
            lam_size_total = lam_size_x * lam_size_y
            anom_scores_to_add = [
                float(next(lam_file).strip()) for _ in range(lam_size_total)
            ]
            anom_scores += anom_scores_to_add
            logging.debug("anom_scores now ends with %s", anom_scores[-4:])

            label_place_in_list = ((label_y_pixel // 64) * lam_size_x) + (
                label_x_pixel // 64
            )
            logging.debug(
                "label_place_in_list is %s of %s", label_place_in_list, lam_size_total
            )
            labels_to_add = [0] * lam_size_total
            labels_to_add[label_place_in_list] = 1

            labels += labels_to_add
            anom_scores_positives[
                (lroc_id, label_x_pixel, label_y_pixel)
            ] = anom_scores_to_add[label_place_in_list]

    anom_scores_positives_list = [
        anom_scores_positives[key] for key in anom_scores_positives
    ]
    logging.info("len(anom_scores, len(labels) is %s", (len(anom_scores), len(labels)))
    logging.info("anom_scores_positives is %s", anom_scores_positives)
    assert len(anom_scores) == len(labels)
    return anom_scores, anom_scores_positives, anom_scores_positives_list, labels


def make_kde_plot(anom_scores, labels, feature_str, show_kde_plot=False):
    """Make a kde plot from anom_scores, labels, and feature_str."""

    xlarge_fontsize = 24
    large_fontsize = 20
    medium_fontsize = 18
    small_fontsize = 14

    # Make a seaborn kde plot of all anom_scores.
    ax = sns.kdeplot(
        anom_scores,
        shade=True,
        label="All samples, background plot",
        gridsize=1000,
        alpha=0.5,
    )

    ax.set(
        xlabel="Anomaly Score",
        ylabel="Density",
        title=f"{feature_str}".title(),
    )

    # Get positive samples.
    anom_scores_pos = [
        anom_score for anom_score, label in zip(anom_scores, labels) if label == 1
    ]
    logging.info("anom_scores_pos is %s", anom_scores_pos)

    # Make a seaborn kde plot of anom_scores_pos.
    ax = sns.kdeplot(
        anom_scores_pos,
        shade=True,
        label="Positive samples, foreground plot",
        gridsize=400,
        alpha=0.5,
    )

    # Make rugplot of positive samples.
    ax = sns.rugplot(
        anom_scores_pos,
        color="purple",
        label="Positive samples, lower rugplot",
        ax=ax,
        height=0.05,
        alpha=1,
    )

    # Set the xlim to just hug right of anom_scores_pos.
    ax.set_xlim(-10, np.max(anom_scores_pos) + 40)

    # Make title, x, y axes bigger.
    ax.title.set_size(xlarge_fontsize)
    ax.xaxis.label.set_size(large_fontsize)
    ax.yaxis.label.set_size(large_fontsize)

    # Make title, x, y axes labels bold.
    ax.title.set_weight("bold")
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_weight("bold")

    # Make legend bigger.
    plt.rc("legend", fontsize=16)

    # Set the legend to include both the kde plot and the rug plot.
    ax.legend(loc="upper right")

    # Save, show, close.
    save_fp = os.path.join(
        "results", f"kde_plot_{feature_str}.png"
    )

    plt.savefig(save_fp, bbox_inches="tight")
    logging.info("Saved kde plot to %s", save_fp)

    if show_kde_plot:
        plt.show()

    plt.close()


def make_ks_test(sample_1, sample_2, feature_str):
    """Make a Kolmogorov-Smirnov test from sample_1 and sample_2."""
    ks_statistic, p_value = stats.ks_2samp(sample_1, sample_2)
    logging.info(
        "Kolmogorov-Smirnov test for %s: ks_statistic is %s, p_value is %s",
        feature_str,
        ks_statistic,
        p_value,
    )
    return ks_statistic, p_value


def make_ks_statistics_table(
    labels_fp,
    model_fp,
    feature_str,
    download_raws_bool,
    inference_bool,
    make_pr_curve_bool,
    make_kde_plot_bool,
    make_ks_test_bool,
    make_ks_statistics_table_bool,
    make_first_detection_statistics_bool,
    make_feature_positive_patches_bool,
    top_n,
    make_top_anom_patches_grid_bool,
    make_random_anom_patches_grid_bool,
):
    """Make a table of Kolmogorov-Smirnov statistics from feature_strs.

    Args:
        labels_fp (str): Labels file path.
        model_fp (str): Model file path.
        feature_str (str): Mission string.
        download_raws_bool (bool): Whether to download raws.
        inference_bool (bool): Whether to run inference.
        make_pr_curve_bool (bool): Whether to make PR curve.
        make_kde_plot_bool (bool): Whether to make kde plot.
        make_ks_test_bool (bool): Whether to do Kolmogorov-Smirnov test.
        make_ks_statistics_table_bool (bool): Whether to make a KS statistics table.
        make_first_detection_statistics_bool (bool): Whether to make first detection statistics.
        make_feature_positive_patches_bool (bool): Whether to make feature positive patches.
        top_n (int): Number of top anomalous patches to make a grid of.
        make_top_anom_patches_grid_bool (bool): Whether to make top anomalous patches grid.
        make_random_anom_patches_grid_bool (bool): Whether to make random anomalous patches grid.

    Returns:
        None.
    """
    logging.info("Starting make_ks_statistics_table method.")
    logging.info("labels_fp is %s", labels_fp)

    ks_test_statistics, p_values = [], []

    # Iterate through each feature, making a table of Kolmogorov-Smirnov
    # statistics. For each feature_str in feature_strs, call the main method.
    feature_strs = sorted(
        [
            "ap12",
            "ap15",
            "ap16",
            "ap17",
            "s3",
            "crater",
            "pit",
            "rockfall",
            "squiggle",
            "weird",
        ]
    )
    for feature_str in feature_strs:
        logging.info("feature_str is %s", feature_str)
        # Hack to get labels_fp correctly.
        labels_fp = os.path.join(
            "data", "interim", "metadata", feature_str + "_labels.csv"
        )
        ks_test_statistic, p_value = main(
            labels_fp=labels_fp,
            model_fp=model_fp,
            feature_str=feature_str,
            download_raws_bool=download_raws_bool,
            inference_bool=inference_bool,
            make_pr_curve_bool=False,
            make_kde_plot_bool=False,
            make_ks_test_bool=True,
            make_ks_statistics_table_bool=False,
            make_first_detection_statistics_bool=False,
            make_feature_positive_patches_bool=False,
            top_n=top_n,
            make_top_anom_patches_grid_bool=False,
            make_random_anom_patches_grid_bool=False,
            make_grid_map_bool=False,
            make_tsne_plot_bool=False,
        )

        ks_test_statistics.append(ks_test_statistic)
        p_values.append(p_value)

    # Print out the table of Kolmogorov-Smirnov statistics.
    ks_statistics_table = pd.DataFrame(
        {
            "Mission": feature_strs,
            "Kolmogorov-Smirnov test statistic": ks_test_statistics,
            "p-value": p_values,
        }
    )
    # Add a column for whether the p-value is significant.
    ks_statistics_table["p-value significant (alpha = 0.05)?"] = (
        ks_statistics_table["p-value"] < 0.05
    )
    logging.info("ks_statistics_table is:\n%s", ks_statistics_table)

    # Also print out the dataframe of statistics in a .csv file.
    ks_statistics_table_fp = os.path.join(
        "results", "ks_statistics_table.csv"
    )
    ks_statistics_table.to_csv(ks_statistics_table_fp, index=False)
    logging.info("Saved ks_statistics_table to %s", ks_statistics_table_fp)

    # Also print out the dataframe of statistics in a .tex file.
    ks_statistics_table_tex_fp = os.path.join(
        "results", "ks_statistics_table.tex"
    )
    ks_statistics_table.to_latex(ks_statistics_table_tex_fp, index=False)
    logging.info("Saved ks_statistics_table to %s", ks_statistics_table_tex_fp)

    return ks_test_statistics, p_values


def make_precision_recall_curve(
    anom_scores, labels, feature_str, show_pr_curve_bool=False
):
    """Make a precision-recall curve from anom_scores, labels, and feature_str."""
    prc = PrecisionRecallCurve(
        labels,
        anom_scores,
        pos_label=1,
        feature_str=feature_str,
        save_to_disk_fp=os.path.join(
            "results", f"pr_curve_feature_{feature_str}"
        ),
    )
    prc.make_plot(show=False, verbose=False)
    return prc


def make_first_detection_statistics(anom_scores, labels):
    """Make first detection statistics from anom_scores, labels, and feature_str.
     Expected first detection idx with random sampling is
    (total number imgs) * 1 / (number_positives + 1), when sampling with replacement.
    """
    d = dict(zip(anom_scores, labels))
    first_detection_idx = 0
    for score in sorted(d, reverse=True):
        if d[score] == 1:
            logging.info("First detection at image index %s", first_detection_idx)
            break
        first_detection_idx += 1

    expected_first_detection_random_idx = len(anom_scores) * 1 / (labels.count(1) + 1)
    logging.info(
        "Expected first detection with random sampling is at image index %f",
        expected_first_detection_random_idx,
    )


def get_lroc_raw_patch_from_lroc_id_and_y_x(
    lroc_id,
    y_ul,
    x_ul,
    lroc_raws_save_fp,
    anom_score,
    in_patch_size_from_model=64,
    out_patch_padding_on_each_side=128,
    draw_anomaly_score=True,
    draw_detailed_patch_info=False,
):
    """Get a patch from lroc_id, y_ul, x_ul, lroc_raws_save_fp, anom_score.

    Args:
        lroc_id: str, the LROC id of the image to get the patch from.
        y_ul: int, the y coordinate of the upper left corner of the patch.
        x_ul: int, the x coordinate of the upper left corner of the patch.
        lroc_raws_save_fp: str, the filepath to the directory where the LROC raws are saved.
        anom_score: float, the anomaly score of the patch.
        in_patch_size_from_model: int, the size of the patch that the model inferenced on.
        out_patch_padding_on_each_side: int, the padding on each side of the out patch.
        draw_detailed_patch_info: bool, whether to draw detailed patch info.

    Returns:
        out_patch: PIL.Image.Image, the patch.
    """
    # Early return if the out image exists in our saved disk cache.
    cache_dir = "/tmp/get_lroc_raw_patch_from_lroc_id_and_y_x_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_name = f"{lroc_id}_y_ul_{y_ul}_x_ul_{x_ul}_padding_{out_patch_padding_on_each_side}.png"
    cache_fp = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_fp):
        img = Image.open(cache_fp)
        logging.debug("Loaded patch from cache at %s", cache_fp)
        logging.debug("Returning patch from cache.")
        return img

    img = Image.open(os.path.join(lroc_raws_save_fp, lroc_id + "_pyr.tif"))
    # Method img.crop has input (left, upper, right, lower)
    x_offset, y_offset = in_patch_size_from_model // 2, in_patch_size_from_model // 2
    x_center = x_ul + x_offset
    y_center = y_ul + y_offset

    padding_from_in_patch_size = in_patch_size_from_model // 2
    padding_from_out_patch_size = out_patch_padding_on_each_side
    total_padding = padding_from_in_patch_size + padding_from_out_patch_size

    img = img.crop(
        (
            x_center - total_padding,
            y_center - total_padding,
            x_center + total_padding,
            y_center + total_padding,
        )
    )
    draw = ImageDraw.Draw(img)

    # Draw anomaly score, lroc_id, and y_ul, x_ul on the lower right corner of
    # the image. Use a black rectangle as background to make the text more
    # readable, and use a white font color. Make the background box tight
    # around the text.

    if draw_detailed_patch_info:
        draw.rectangle(
            (
                img.width - 160,
                img.height - 45,
                img.width - 10,
                img.height - 10,
            ),
            fill="black",
        )
        draw.text(
            (img.width - 155, img.height - 45),
            f"Anom score: {anom_score:.2f}",
            fill="white",
        )
        draw.text(
            (img.width - 155, img.height - 35),
            f"LROC ID: {lroc_id}",
            fill="white",
        )
        draw.text(
            (img.width - 155, img.height - 25),
            f"y_ul: {y_ul}, x_ul: {x_ul}",
            fill="white",
        )
    elif draw_anomaly_score:
        # Only draw the anomaly score on the lower right corner of the image.
        # Make anomaly score font bigger to make it easier to read.
        # Font path is /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf

        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, 20)

        if anom_score < 1000:
            draw.rectangle(
                (
                    img.width - 215,
                    img.height - 45,
                    img.width - 10,
                    img.height - 20,
                ),
                fill="black",
            )
            draw.text(
                (img.width - 210, img.height - 45),
                f"Anom score: {anom_score:.2f}",
                fill="white",
                font=font,
            )
        else:
            draw.rectangle(
                (
                    img.width - 225,
                    img.height - 45,
                    img.width - 5,
                    img.height - 20,
                ),
                fill="black",
            )
            draw.text(
                (img.width - 220, img.height - 45),
                f"Anom score: {anom_score:.2f}",
                fill="white",
                font=font,
            )

    # Draw the model input patch in red on the image.
    draw.rectangle(
        (
            out_patch_padding_on_each_side,
            out_patch_padding_on_each_side,
            out_patch_padding_on_each_side + in_patch_size_from_model,
            out_patch_padding_on_each_side + in_patch_size_from_model,
        ),
        outline="red",
    )

    # Cache the image to disk to avoid the long time it takes to draw the
    # image. Save to cache_fp.
    img.save(cache_fp)
    logging.info(f"Saved cache image to {cache_fp}")

    return img


def make_pil_mosaic(
    feature_str,
    images,
    save_fp_top_or_random="",
    show_bool=False,
):
    """Make a PIL mosaic of n images."""
    # Make a square grid mosaic of the top n images.
    widths, heights = zip(*(i.size for i in images))
    # Assumption: All widths, heights are the same.
    width, height = widths[0], heights[0]
    total_height = height * math.ceil(math.sqrt(len(images)))
    total_width = width * math.ceil(math.sqrt(len(images)))

    new_im = Image.new("RGB", (total_width, total_height))
    x_offset = 0
    y_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
        if x_offset >= total_width:
            x_offset = 0
            y_offset += im.size[1]

    # Cut off bottom of image, due to a bottom black row,
    # if decide to.
    cut = False
    if cut:
        new_im = new_im.crop((0, 0, total_width, total_height - height))

    out_mosaic_fp = os.path.join(
        "results",
        f"feature_{feature_str}_{save_fp_top_or_random}_{len(images)}_patches_grid.png",
    )
    new_im.save(out_mosaic_fp)
    logging.info(
        "Saved PIL mosaic of %s images to %s",
        len(images),
        out_mosaic_fp,
    )
    if show_bool:
        new_im.show()


def make_feature_positive_patches_sorted_by_anom_score(
    feature_str,
    anom_scores_positives,
    label_lines,
    lroc_raws_save_fp,
    num_patches_to_draw=32,
    make_feature_positive_patches_bool=True,
):
    """Make feature positive patches sorted by anom_score from
    anom_scores_positives, feature_str, label_lines, and lroc_raws_save_fp.
    """
    if not make_feature_positive_patches_bool:
        logging.info(
            "Not making feature positive patches sorted"
            "by anomaly scores, since make_feature_positive_patches_bool"
            "was False."
        )
        return None

    # Make a PIL mosaic of the top n images.
    positive_imgs = []

    for label_line in label_lines:
        lroc_id = label_line[0]
        y_label = int(label_line[2])
        x_label = int(label_line[1])
        anom_score = anom_scores_positives[(lroc_id, x_label, y_label)]
        logging.info(
            "lroc_id, y_label, x_label, anom_score are %s",
            (lroc_id, y_label, x_label, anom_score),
        )

        # Convert y, x to y_ul, x_ul, based on a 64x64 grid.
        y_ul = (y_label // 64) * 64
        x_ul = (x_label // 64) * 64
        logging.info("y_ul, x_ul are %s", (y_ul, x_ul))

        img = get_lroc_raw_patch_from_lroc_id_and_y_x(
            lroc_id, y_ul, x_ul, lroc_raws_save_fp, anom_score
        )
        positive_imgs.append([img, anom_score])

    logging.info("positive_imgs is: %s", positive_imgs)
    positive_imgs.sort(key=lambda x: x[1], reverse=True)
    positive_imgs = [positive_image[0] for positive_image in positive_imgs]
    positive_imgs = positive_imgs[:num_patches_to_draw]

    make_pil_mosaic(
        feature_str,
        positive_imgs,
        save_fp_top_or_random="top_feature_positive",
        show_bool=False,
    )


def get_top_n_and_random_n_lines(
    feature_str,
    top_n,
) -> List[Tuple[str, int, int, float, float, float]]:
    """Make sorted top anom patches list, from feature_str.
    Each out line should have the format: lroc_id, y_ul, x_ul, lat, long, anom_score
    Also make random n lines, for a useful comparison.
    """
    logging.info("Getting top n and random n lines from feature_str %s", feature_str)
    logging.info("top_n is %s", top_n)

    lam_dir = os.path.join("data", "processed")

    csv_fp = os.path.join(lam_dir, "lams_to_csvs", f"lams_to_csv_{feature_str}.csv")
    if not os.path.exists(csv_fp):
        csv_fp = lam_to_csv(feature_str=feature_str, lam_dir=lam_dir, out_csv_fp=csv_fp)

    # Open out_csv_fp, sort all lines by anom_score, and return the top_n lines.
    with open(csv_fp, "r", encoding="utf-8") as f:
        logging.info("Reading lines from %s", csv_fp)
        lines = f.readlines()
        lines = [line.strip().split(",") for line in lines]

        # Find the indices of the lroc_id, x_ul, y_ul, anom_score columns.
        # Also find the indices of long, lat.
        lroc_id_col_idx = lines[0].index("lroc_id")
        x_col_idx = lines[0].index("x_ul")
        y_col_idx = lines[0].index("y_ul")
        long_col_idx = lines[0].index("long")
        lat_col_idx = lines[0].index("lat")
        anom_score_col_idx = lines[0].index("anom_score")

        # Set lines to be all lines except the header.
        lines = lines[1:]

        # Filter out lines with x <= 64, left black strip edge of image.
        # Filter out lines with x >= 4992, right black strip edge of image.
        logging.info("Filtering out lines with x <= 64, x >= 4992.")
        lines = [
            line
            for line in lines
            if int(line[x_col_idx]) > 64 and int(line[x_col_idx]) < 4992
        ]

        # Sort lines by anom_score, descending.
        logging.info("Sorting lines by anom_score, descending.")
        lines.sort(key=lambda x: float(x[anom_score_col_idx]), reverse=True)
        # Assign the top_n lines.
        top_n_lines = [
            (
                str(line[lroc_id_col_idx]),
                int(line[y_col_idx]),
                int(line[x_col_idx]),
                float(line[lat_col_idx]),
                float(line[long_col_idx]),
                float(line[anom_score_col_idx]),
            )
            for line in lines[1 : top_n + 1]
        ]
        # Assign also random n lines, useful as a comparison and sanity check.
        random_n_lines = [
            (
                str(line[lroc_id_col_idx]),
                int(line[y_col_idx]),
                int(line[x_col_idx]),
                float(line[lat_col_idx]),
                float(line[long_col_idx]),
                float(line[anom_score_col_idx]),
            )
            for line in random.sample(lines, top_n)
        ]

    # Log header and each line of top_n_lines.
    header = "lroc_id,y_ul,x_ul,lat,long,anom_score"
    logging.info("top_n_lines header: %s", header)
    for line in top_n_lines:
        logging.debug(
            "%s, %s, %s, %s, %s, %s",
            line[0],
            line[1],
            line[2],
            f"{line[3]:.4f}",
            f"{line[4]:.4f}",
            f"{line[5]:.4f}",
        )

    # Save the top_n_lines to a csv.
    top_n_lines_csv_fp = os.path.join(
        "results",
        f"feature_{feature_str}_top_{top_n}_patches.csv",
    )
    with open(top_n_lines_csv_fp, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for line in top_n_lines:
            f.write(
                f"{line[0]},{line[1]},{line[2]},{line[3]:.3f},{line[4]:.3f},{line[5]:.2f}\n"
            )
    logging.info(
        "Saved top_n_lines to %s",
        top_n_lines_csv_fp,
    )

    return top_n_lines, random_n_lines


def return_top_images(
    top_n_lines,
    lroc_raws_save_fp,
    special_mode=None,
):
    """Return the top images for each feature.

    Args:
        top_n_lines: List of tuples, each tuple is a line from the top_n_lines csv.
        lroc_raws_save_fp: String, filepath to the lroc_raws directory.
        special_mode: String, "tsne" for use in tsne mode, None otherwise.

    Returns:
        List of PIL.Image.Image objects, each image is a top image.
    """
    top_images = []
    in_patch_size_from_model = 64
    for i, line in tqdm.tqdm(enumerate(top_n_lines)):
        logging.debug("line is %s", line)
        lroc_id, y_ul, x_ul, _, _, anom_score, positive = line
        logging.debug("Working on lroc_id %s, %s of %s.", lroc_id, i, len(top_n_lines))

        if special_mode == "tsne":
            img = get_lroc_raw_patch_from_lroc_id_and_y_x(
                lroc_id,
                y_ul,
                x_ul,
                lroc_raws_save_fp,
                anom_score,
                in_patch_size_from_model=64,
                out_patch_padding_on_each_side=64,
                draw_anomaly_score=False,
                draw_detailed_patch_info=False,
            )

        else:
            img = get_lroc_raw_patch_from_lroc_id_and_y_x(
                lroc_id,
                y_ul,
                x_ul,
                lroc_raws_save_fp,
                anom_score,
                in_patch_size_from_model=64,
                out_patch_padding_on_each_side=128,
                draw_anomaly_score=True,
                draw_detailed_patch_info=False,
            )
        # Workaround for OSError: Errno 24 Too many open files.
        keep = img.copy()
        top_images.append(keep)
        img.close()

    return top_images


def make_top_anom_patches_grid(
    feature_str,
    top_n_lines,
    random_control=False,
    make_top_anom_patches_grid_bool=True,
    show_top_anom_patches_grid_bool=False,
):
    """Make a grid of the top anomalous patches for a given feature,
    possibly from multiple lroc raw images.
    """
    lroc_raws_save_fp = os.path.join("data", "raw", feature_str, "class_0")
    if not random_control:
        logging.info("Making a grid of the top anomalous patches for a given feature.")
    else:
        logging.info("Making a grid of random patches for a given feature.")
    logging.info("lroc_raws_save_fp is %s", lroc_raws_save_fp)

    # Make a grid of the top n images, likely from multiple lroc raw images.
    if make_top_anom_patches_grid_bool:

        top_images = return_top_images(top_n_lines, lroc_raws_save_fp)
        save_fp_top_or_random = "top_anomalies" if not random_control else "random"
        make_pil_mosaic(
            feature_str,
            top_images,
            save_fp_top_or_random=save_fp_top_or_random,
            show_bool=show_top_anom_patches_grid_bool,
        )


def images_scatter(x, y, images, ax=None, zoom=0.1, reverse=False):
    """Helper function for making tsne image scatter plots.

    Args:
        x: List of floats, x coordinates.
        y: List of floats, y coordinates.
        images: List of PIL.Image.Image objects, each image is a top image, or
        a list of numpy arrays, each array is a top image?
        ax: Matplotlib axis object.
        zoom: Float, zoom factor for the images?
        reverse: Boolean, whether to reverse the order of the images.

    Returns:
        List of matplotlib offset image objects?
    """
    artists = []
    x, y = np.atleast_1d(x, y)

    if reverse:
        images, x, y = list(reversed(images)), list(reversed(x)), list(reversed(y))

    for i, img in tqdm.tqdm(enumerate(images)):
        img = OffsetImage(img, zoom=zoom, cmap="gray")
        ab = AnnotationBbox(img, (x[i], y[i]), xycoords="data", frameon=False)
        artists.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def add_positives_column(top_n_lines, feature_str):
    """Adds a column of 1's to the top_n_lines csv exactly when
    the patch is a positive (anomalous) patch.

    """

    logging.info("top_n_lines[0]: %s", top_n_lines[0])
    logging.info("top_n_lines[-1]: %s", top_n_lines[-1])
    logging.debug("top_n_lines: %s", top_n_lines)
    logging.info("len(top_n_lines): %s", len(top_n_lines))

    # Open label file associated with our feature string.
    label_fp = os.path.join("data", "interim", "metadata", feature_str + "_labels.csv")
    logging.info("label_fp is %s", label_fp)
    with open(label_fp, "r") as f:
        reader = csv.reader(f)
        label_lines = list(reader)
    logging.info("label_lines[0] is %s", label_lines[0])
    logging.info("label_lines[1] is %s", label_lines[1])
    logging.info("label_lines is %s", label_lines)

    # Add a column of 1's to the top_n_lines csv exactly when
    # the patch is a positive (anomalous) patch.
    top_n_lines_with_positives_column = []
    for top_n_line in top_n_lines:
        top_n_line = list(top_n_line)
        logging.debug("top_n_line is %s", top_n_line)

        # Search whether the 64 by 64 patch determined by the top upper left
        # coordinates y_ul, x_ul in top_n_line is a positive patch. Search first
        # for a match with lroc_ids, and then whether any of the coordinates in
        # the label_lines list fall in the 64 by 64 patch determined by y_ul,
        # x_ul from top_n_line. Each top_n_line has format lroc_id, y_ul, x_ul, lat, long,
        # anom_score, while each label_line has format 'lroc_id', 'x', 'y'.

        # Search for a match with lroc_ids.
        lroc_id_top_n_line = top_n_line[0]
        for label_line in label_lines:
            # Skip the header line.
            if label_line[1] == "x":
                continue
            lroc_id_label_line = label_line[0]
            if lroc_id_top_n_line == lroc_id_label_line:
                logging.debug(
                    "Found a match in lroc_ids betweeen %s and %s",
                    lroc_id_top_n_line,
                    lroc_id_label_line,
                )
                # Now check if there is a match in coordinates when matching lroc_ids.
                y_ul_top_n_line = int(top_n_line[1])
                x_ul_top_n_line = int(top_n_line[2])
                y_label_line = float(label_line[2])
                x_label_line = float(label_line[1])
                if (
                    x_ul_top_n_line <= x_label_line <= x_ul_top_n_line + 64
                    and y_ul_top_n_line <= y_label_line <= y_ul_top_n_line + 64
                ):
                    logging.info(
                        "FOUND A MATCH IN COORDINATES, with y_label_line, x_label_line %s "
                        "IN the 64 x 64 patch determined by y_ul_top_n_line, x_ul_top_n_line %s",
                        (y_label_line, x_label_line),
                        (y_ul_top_n_line, x_ul_top_n_line),
                    )
                    # Append a 1 to the top_n_line tuple
                    top_n_line.append(1)
                    time.sleep(0.01)
                    break
                else:
                    logging.debug(
                        "Found no match in coordinates, with y_label_line, x_label_line %s "
                        "not in the 64 x 64 patch determined by y_ul_top_n_line, x_ul_top_n_line %s",
                        (y_label_line, x_label_line),
                        (y_ul_top_n_line, x_ul_top_n_line),
                    )
        # If we didn't find a match after all label_lines, then this is a negative patch.
        # Append a 0 to the top_n_line tuple
        if len(top_n_line) == 6:
            top_n_line.append(0)
        top_n_lines_with_positives_column.append(tuple(top_n_line))

    logging.info(
        "top_n_lines_with_positives_column[0] is %s",
        top_n_lines_with_positives_column[0],
    )
    logging.debug(
        "top_n_lines_with_positives_column is %s",
        top_n_lines_with_positives_column,
    )
    return top_n_lines_with_positives_column


def make_tsne_plot(
    feature_str,
    top_n_lines,
    lam_fp="data/processed/",
    lroc_raws_fp="data/raw/all/class_0",
    top_n=64,
    tsne_pca_random_state=0,
):
    """Makes a t-sne plot of the top_n most anomalous patches.

    Args:
        lam_fp (str): The filepath to directory containing the .lam files.
        top_n (int): The number of most anomalous patches to plot. Default is 64.

    Returns:
        None
    """

    top_images = return_top_images(
        top_n_lines,
        lroc_raws_fp,
        special_mode="tsne",
    )
    logging.info("len(top_images): %s", len(top_images))
    logging.info("Making a t-sne plot of top_images.")
    logging.info("top_n: %s", top_n)
    logging.info("tsne_pca_random_state: %s", tsne_pca_random_state)

    early_exaggeration = 12.0
    logging.info("early_exaggeration: %s", early_exaggeration)
    # Auto-set of learning rate based on a published heuristic.
    # From sklearn docs: If the learning rate is too high, the data may look
    # like a ‘ball’ with any point approximately equidistant from its nearest
    # neighbours. If the learning rate is too low, most points may look
    # compressed in a dense cloud with few outliers.
    # I have actually seen this the other way around, where the learning rate
    # is too low and the data looks like a ball, while if the learning rate
    # is too high, the data looks compressed in a dense cloud with few outliers.
    # Sklearn implementation of auto-set of learning rate.
    learning_rate = round(max(top_n / early_exaggeration / 4.0, 50.0), 2)
    logging.info("learning_rate: %s", learning_rate)

    perplexity = top_n / 100.0
    logging.info("perplexity set to: %s", perplexity)

    tsne = TSNE(
        random_state=tsne_pca_random_state,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        perplexity=perplexity,
        n_components=2,
        n_jobs=-1,
        verbose=9,
        method="barnes_hut",
    )

    # Load top_images, an array of PIL.Image.Image, into a format suitable for
    # sklearn.manifold.TSNE. Make of shape (n_samples, n_features), where
    # n_samples is the number of images, and n_features is the number of pixels
    # in each image.
    top_images_array = np.array([np.array(img) for img in top_images])
    logging.info("top_images_array.shape: %s", top_images_array.shape)
    reshaped_top_images_array = top_images_array.reshape(
        top_images_array.shape[0],
        -1,
    )
    logging.info("reshaped_top_images_array.shape: %s", reshaped_top_images_array.shape)

    # Reduce the dimensionality of the top_images_array using PCA.
    n_components = min(64, len(top_images_array))
    pca = PCA(n_components=n_components, random_state=tsne_pca_random_state)
    pca.fit(reshaped_top_images_array)
    top_images_array_pca = pca.transform(reshaped_top_images_array)
    logging.info("pca.n_components_: %s", pca.n_components_)
    logging.debug("pca.explained_variance_ratio_: %s", pca.explained_variance_ratio_)
    logging.info(
        "pca.explained_variance_ratio_.sum(): %s", pca.explained_variance_ratio_.sum()
    )
    logging.debug("pca.singular_values_: %s", pca.singular_values_)
    logging.info("pca.mean_.shape: %s", pca.mean_.shape)
    logging.info("pca.mean_: %s", pca.mean_)
    logging.info("top_images_array_pca.shape: %s", top_images_array_pca.shape)

    # Fit t-sne
    logging.info("Fitting tsne to top_images_array_pca.")
    tsne_results = tsne.fit_transform(top_images_array_pca)
    logging.info("tsne_results.shape: %s", tsne_results.shape)

    # Plot the t-sne.
    logging.info(
        "Plotting the resulting t-sne, with the images located in top_images_array."
    )
    figsize = (8.5, 11)
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter background points.
    ax.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        alpha=0.5,
        s=0.5,
        c="black",
    )

    # Go through each image in images, drawing a colored
    # border if the cooresponding last entry of
    # top_n_lines is a 1.
    for i, img in enumerate(top_images):
        if top_n_lines[i][-1] == 1:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            draw.rectangle(
                [
                    (0, 0),
                    (img.size[0] - 1, img.size[1] - 1),
                ],
                outline="orange",
                width=4,
            )
            top_images[i] = img

    # Scatter the top most anomalous patches.
    top_n_images_scatter_limit = 4096
    images_scatter(
        tsne_results[:top_n_images_scatter_limit, 0],
        tsne_results[:top_n_images_scatter_limit, 1],
        images=top_images[:top_n_images_scatter_limit],
        ax=ax,
        zoom=0.1,
        reverse=True,
    )

    # Hide the axes, everything except the images.
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Save the figure.
    dpi = 600
    save_name = (
        f"tsne_"
        f"feat_{feature_str}_"
        f"top_{top_n}_"
        f"early_ex_{early_exaggeration:.2f}_"
        f"lr_{learning_rate:.2f}_"
        f"perp_{perplexity:.2f}_"
        f"state_{tsne_pca_random_state}_"
        f"dpi_{dpi}_"
        f"time_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.png"
    )
    save_fp = os.path.join(
        "results",
        save_name,
    )
    plt.savefig(save_fp, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    logging.info("Saved t-sne plot to %s", save_fp)
    plt.close()

    return None


def make_grid_map(
    feature_str,
    label_lines,
    pixel_expand_factor=64,
    lam_file_order_to_consider=0,
):
    """Make a grid map visualizing the anomaly scores for a given feature.

    Args:

        feature_str (str): The feature string, e.g. "crater".
        label_lines (list): A list of lines from the label csv.
        pixel_expand_factor (int): The factor by which to expand the pixels. Default is 64.
        Use 64 to see the anomaly grid map at the resolution of the .tif input file.
        Use 1 for no expansion and to see the anomaly grid map at the original resolution.
        lam_file_order_to_consider (int): The lam file to consider. Default is 0.

    Returns:

        None
    """
    # Simplification: for now we only get the lam_file_order_to_consider^th lam
    # file in our label list.

    lroc_id = label_lines[lam_file_order_to_consider][0]
    logging.info("lroc_id is %s", lroc_id)

    lam_fp = os.path.join(
        "data",
        "processed",
        feature_str,
        f"{lroc_id}.lam",
    )
    logging.info("lam_fp is %s", lam_fp)

    with open(lam_fp, "r", encoding="utf-8") as f:
        lam_lines = f.readlines()
    logging.info("Read in %s lines from %s", len(lam_lines), lam_fp)
    logging.info("First 16 lines of the .lam file:")
    for line in lam_lines[:16]:
        logging.info(line.strip())

    tif_y_max = int(float(lam_lines[1].strip()))
    tif_x_max = int(float(lam_lines[2].strip()))
    lam_y_max = int(float(lam_lines[5].strip()))
    lam_x_max = int(float(lam_lines[6].strip()))
    logging.info("tif_y_max, tif_x_max are %s, %s", tif_y_max, tif_x_max)
    logging.info("lam_y_max, lam_x_max are %s, %s", lam_y_max, lam_x_max)

    grid_map = [float(x) for x in lam_lines[9:]]
    logging.info("First values of grid_map:")
    for value in grid_map[:4]:
        logging.info(value)
    assert len(grid_map) == lam_y_max * lam_x_max

    grid_map = np.array(grid_map).reshape(lam_y_max, lam_x_max)
    logging.info("grid_map.shape is %s", grid_map.shape)

    # Expand the pixels by a factor of pixel_expand_factor.
    grid_map = np.repeat(
        np.repeat(grid_map, pixel_expand_factor, axis=0),
        pixel_expand_factor,
        axis=1,
    )
    assert grid_map.shape == (
        lam_y_max * pixel_expand_factor,
        lam_x_max * pixel_expand_factor,
    )
    logging.info(
        "grid_map.shape after expanding the pixels is %s",
        grid_map.shape,
    )
    logging.info("First value of grid_map:")
    for value in grid_map[:1]:
        logging.info(value)

    # Positive coordinates
    positive_coords = []
    for line in label_lines:
        if line[0] == lroc_id:
            positive_coords.append((int(line[1]), int(line[2])))
    logging.info("positive_coords is %s", positive_coords)

    positive_coords_decimals = []
    for x_label, y_label in positive_coords:
        positive_coords_decimals.append((x_label / tif_x_max, y_label / tif_y_max))
    logging.info("positive_coords_decimals is %s", positive_coords_decimals)

    # Plot the grid map.
    figsize = (4, 8)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid_map, cmap="Reds")
    marker = (4, 2, 0)

    for x_label, y_label in positive_coords_decimals:
        ax.scatter(
            x_label * grid_map.shape[1],
            y_label * grid_map.shape[0],
            marker=marker,
            s=80,
            linewidths=0.2,
            c="black",
        )

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Anomaly Score", rotation=-90, va="bottom")

    fig.tight_layout()
    save_path = os.path.join(
        "results",
        f"anomaly_map_{lroc_id}_in_{feature_str}_set.png",
    )
    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=600,
    )
    logging.info("Saved grid map to %s", save_path)
    plt.close()

    return None


def main(
    labels_fp,
    model_fp,
    feature_str,
    download_raws_bool,
    inference_bool,
    make_pr_curve_bool,
    make_kde_plot_bool,
    make_ks_test_bool,
    make_ks_statistics_table_bool,
    make_first_detection_statistics_bool,
    make_feature_positive_patches_bool,
    top_n,
    make_top_anom_patches_grid_bool,
    make_random_anom_patches_grid_bool,
    make_grid_map_bool,
    make_tsne_plot_bool,
):
    """Main method."""

    logging.info("Starting main method.")
    logging.info("labels_fp is %s", labels_fp)
    logging.info("feature_str is %s", feature_str)

    if make_tsne_plot_bool:
        for top_n in [top_n]:  # [256, 512, 1024, 2048, 4096, 8192]:
            top_n_lines, _ = get_top_n_and_random_n_lines(feature_str, top_n=top_n)
            top_n_lines_with_positives_column = add_positives_column(
                top_n_lines, feature_str
            )
            for random_state in range(0, 4):
                make_tsne_plot(
                    feature_str,
                    top_n_lines_with_positives_column,
                    top_n=top_n,
                    tsne_pca_random_state=random_state,
                )
        return None

    if make_top_anom_patches_grid_bool:
        top_n_lines, _ = get_top_n_and_random_n_lines(feature_str, top_n=top_n)
        make_top_anom_patches_grid(
            feature_str,
            top_n_lines,
        )
        return None

    if make_random_anom_patches_grid_bool:
        _, random_n_lines = get_top_n_and_random_n_lines(feature_str, top_n=top_n)
        make_top_anom_patches_grid(
            feature_str,
            random_n_lines,
            random_control=True,
        )
        return None

    lroc_raw_ids_list, label_lines = get_label_lines_lroc_raw_ids_lists(labels_fp)
    out_lroc_raws_urls_fp = write_out_lroc_raw_urls(feature_str, lroc_raw_ids_list)

    lroc_raws_save_fp = download_lroc_raws(
        feature_str, out_lroc_raws_urls_fp, download_raws_bool
    )
    out_inference_fp = inference_on_lroc_raws(
        feature_str, lroc_raws_save_fp, model_fp, inference_bool
    )

    if not download_raws_bool:
        (
            anom_scores,
            anom_scores_positives,
            anom_scores_positives_list,
            labels,
        ) = generate_anom_scores_and_labels(label_lines, out_inference_fp)

    if make_pr_curve_bool:
        make_precision_recall_curve(
            anom_scores, labels, feature_str, show_pr_curve_bool=True
        )

    if make_kde_plot_bool:
        make_kde_plot(anom_scores, labels, feature_str)

    if make_ks_test_bool:
        sample_1, sample_2 = anom_scores, anom_scores_positives_list
        ks_test_statistic, p_value = make_ks_test(sample_1, sample_2, feature_str)

    if make_ks_statistics_table_bool:
        make_ks_statistics_table(
            labels_fp,
            model_fp,
            feature_str,
            download_raws_bool,
            inference_bool,
            make_pr_curve_bool,
            make_kde_plot_bool,
            make_ks_test_bool,
            make_ks_statistics_table_bool,
            make_first_detection_statistics_bool,
            make_feature_positive_patches_bool,
            top_n,
            make_top_anom_patches_grid_bool,
            make_random_anom_patches_grid_bool,
        )

    if make_first_detection_statistics_bool:
        make_first_detection_statistics(anom_scores, labels)

    if make_feature_positive_patches_bool:
        make_feature_positive_patches_sorted_by_anom_score(
            feature_str,
            anom_scores_positives,
            label_lines,
            lroc_raws_save_fp,
            num_patches_to_draw=top_n,
            make_feature_positive_patches_bool=make_feature_positive_patches_bool,
        )

    if make_ks_test_bool:
        return ks_test_statistic, p_value

    if make_grid_map_bool:
        make_grid_map(
            feature_str,
            label_lines,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate module to evaluate models.")
    parser.add_argument(
        "--feature_str",
        "-f",
        type=str,
        default="ap17",
        help=(
            "Feature string to evaluate, e.g. 'ap12', 's3', 'crater', 'pit'"
            " (default: ap17)."
        ),
    )

    parser.add_argument(
        "--download_raws",
        "-d",
        action="store_true",
        help="Whether to download raws (default: False).",
    )

    parser.add_argument(
        "--inference",
        "-i",
        action="store_true",
        help="Whether to run inference (default: False).",
    )

    parser.add_argument(
        "--model_fp",
        type=str,
        default=(
            "models/train1_arts_exp_time_1646954340"
            "__stage_iii_train1_squares_2K_debug_models_model_eoe_5.pt"
        ),
        help=(
            "Model file path (default: models/train1_arts_exp_time_1646954340"
            "__stage_iii_train1_squares_2K_debug_models_model_eoe_5.pt)."
        ),
    )

    parser.add_argument(
        "--labels_home",
        type=str,
        default="data/interim/metadata",
        help="Labels home (default: data/interim/metadata).",
    )

    parser.add_argument(
        "--log_filename",
        type=str,
        default="logs/evaluate_{int(time.time())}.log",
        help="Log filename (default: logs/evaluate_{int(time.time())}.log).",
    )

    parser.add_argument(
        "--top-n",
        "-n",
        type=int,
        default=4,
        help="Number of patches to make a grid of, if making a grid (default: 4).",
    )

    parser.add_argument(
        "--make_first_detection_statistics",
        action="store_true",
        help="Whether to make first detection statistics (default: False).",
    )

    parser.add_argument(
        "--make_pr_curve",
        "-c",
        action="store_true",
        help="Whether to make PR curve (default: False).",
    )

    parser.add_argument(
        "--make_kde_plot",
        "-k",
        action="store_true",
        help="Whether to make kde plot (default: False).",
    )

    parser.add_argument(
        "--make_ks_test",
        "-s",
        action="store_true",
        help="Whether to do Kolmogorov-Smirnov test. This test is used to compare"
        " whether our two empirical samples, anomaly scores and anomaly scores of positive,"
        " come from the same distribution (default: False).",
    )

    parser.add_argument(
        "--make_ks_statistics_table",
        action="store_true",
        help="Whether to make a KS statistics table (default: False).",
    )

    parser.add_argument(
        "--make_top_anom_patches_grid",
        "-a",
        action="store_true",
        help="Whether to make top anomalous patches grid (default: False).",
    )

    parser.add_argument(
        "--make_random_anom_patches_grid",
        "-r",
        action="store_true",
        help="Whether to make random anomalous patches grid (default: False).",
    )

    parser.add_argument(
        "--make_feature_positive_patches",
        "-p",
        action="store_true",
        help="Whether to make top feature positive patches (default: False).",
    )

    parser.add_argument(
        "--make_grid_map",
        "-g",
        action="store_true",
        help="Whether to make a grid map based on anomaly scores (default: False).",
    )

    parser.add_argument(
        "--make_tsne_plot",
        "-t",
        action="store_true",
        help="Whether to make a t-sne plot of top anomalous patches (default: False).",
    )

    # Setup
    args = parser.parse_args()
    labels_filepath = os.path.join(args.labels_home, args.feature_str + "_labels.csv")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.log_filename),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("args are %s", args)
    logging.info("labels_filepath is %s", labels_filepath)
    logging.info("args.model_fp is: %s", args.model_fp)

    main(
        labels_fp=labels_filepath,
        model_fp=args.model_fp,
        feature_str=args.feature_str,
        download_raws_bool=args.download_raws,
        inference_bool=args.inference,
        make_pr_curve_bool=args.make_pr_curve,
        make_kde_plot_bool=args.make_kde_plot,
        make_ks_test_bool=args.make_ks_test,
        make_ks_statistics_table_bool=args.make_ks_statistics_table,
        make_first_detection_statistics_bool=args.make_first_detection_statistics,
        make_feature_positive_patches_bool=args.make_feature_positive_patches,
        top_n=args.top_n,
        make_top_anom_patches_grid_bool=args.make_top_anom_patches_grid,
        make_random_anom_patches_grid_bool=args.make_random_anom_patches_grid,
        make_grid_map_bool=args.make_grid_map,
        make_tsne_plot_bool=args.make_tsne_plot,
    )
