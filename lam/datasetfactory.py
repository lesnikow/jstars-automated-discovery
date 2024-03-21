# -*- coding: utf-8 -*-
"""datasetfactory module
Define the module that encapsulates attributes and methods related to dataset
creation we use for our lunar anomalies project.

You will first need to download raw images for this script to work properly, by
using e.g.

python3 lam/evaluate.py --feature_str pit --download

or by using the ASU PDS and putting these raw LROC images into the data/raw/
directory.

After this, update in_img_dir below to the directory path of the raw .tiff
files. The default im_img_dir should work if you run the above python3 command
with the "pit" feature_str.

Also note cnt_lim below. The default value 2**2 is low to test this module is
working for you. For real training runs, something like cnt_lim = 2*20 will
generate appropriately sized large training sets, as in the JSTARS paper.
"""
import os
import random
from os import listdir
from os.path import isfile, join

import numpy
import tqdm
from PIL import Image


def main():
    """Main function to generate our image squares dataset."""
    Image.MAX_IMAGE_PIXELS = 5e8

    in_img_dir = f"data/raw/pit/class_0/"
    out_img_dir = "data/processed/train/class_0/"
    dsf = DatasetFactory(
        in_img_dir=in_img_dir,
        out_img_dir=out_img_dir,
        verbose=True,
    )

    dsf.in_img_dir
    in_img_ids = sorted(
        [
            f.split(".")[0]
            for f in listdir(dsf.in_img_dir)
            if isfile(join(dsf.in_img_dir, f))
        ][:]
    )
    print(f"in_imgs_dir is {in_img_dir}")
    print(f"in_img_ids is {in_img_ids}")
    random.shuffle(in_img_ids)

    for in_img_id in tqdm.tqdm(in_img_ids):
        if in_img_id == "":
            continue
        dsf.make_crops_one_img(
            in_img_id=in_img_id, 
            verbose=True
        )


class DatasetFactory:
    def __init__(
        self,
        in_img_dir=None,
        out_img_dir=None,
        verbose=False,
    ):
        # Class attributions
        self.verbose = verbose

        # Input data information.
        self.in_img_dir = in_img_dir

        # Ouput data information.
        self.out_img_dir = out_img_dir
        self.out_width = 64
        self.out_height = 64

    def make_crops_one_img(
        self,
        cnt_lim=None,
        in_img_id=None,
        in_img_ext=".tif",
        out_img_id=None,
        out_img_ext="jpg",
        verbose=None,
        vv=False,
        print_info_interval=2 ** 10,
    ):
        """Makes out image crops for one image."""

        # Defaults in no other input given.
        if in_img_id is None:
            in_img_id = "M102000149RC"
        if cnt_lim is None:
            cnt_lim = 2 ** 1
        if verbose is None:
            verbose = self.verbose

        in_fp = os.path.join(
            self.in_img_dir,
            in_img_id + in_img_ext,
        )
        img = Image.open(in_fp)
        img_width, img_height = img.size
        if self.verbose:
            print("In image width: {}".format(img_width))
            print("In image height: {}\n".format(img_height))

        # Image outs.
        out_img_id = in_img_id
        self.out_fp = os.path.join(self.out_img_dir, out_img_id)

        if not os.path.exists(self.out_fp):
            print("Making out dir {}\n".format(self.out_fp))
            os.makedirs(self.out_fp)
        else:
            print("Notice: out dir {} already exists\n".format(self.out_fp))

        # Each pixel gets used overlap_factor**2 times
        overlap_factor = 1
        imgs_tot = overlap_factor ** 2 * int(
            img_height * img_width / (self.out_width * self.out_height)
        )
        x_lim = (img_width - 32) - self.out_width
        y_lim = img_height - self.out_height
        if verbose:
            print((x_lim, y_lim))
        cnt_lim = min(cnt_lim, imgs_tot)
        cnt = 0

        print("Working on image with id: {}".format(in_img_id))
        for y in tqdm.tqdm(range(0, y_lim, int(self.out_height / overlap_factor))):
            for x in range(32, x_lim, int(self.out_width / overlap_factor)):
                if cnt >= cnt_lim:
                    break
                if cnt % print_info_interval == 0 and verbose:
                    print(
                        "Working on square {}/{} with top-left coords "
                        "x:{},y:{}...".format(cnt, cnt_lim, x, y)
                    )
                bbox = (x, y, x + self.out_width, y + self.out_height)
                if cnt % print_info_interval == 0 and verbose:
                    print(bbox)
                if vv:
                    print(bbox)
                crop = img.crop(bbox)
                out_name = "{:010d}_x{}_y{}.{}".format(cnt, x, y, out_img_ext)
                save_fp = os.path.join(self.out_fp, out_name)
                crop.save(save_fp)
                cnt += 1
        print("Done with working on image with id: {}".format(in_img_id))


if __name__ == "__main__":
    main()
