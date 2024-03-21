# -*- coding: utf-8 -*-
"""data module
Here we define our data module that encapsulates attributes and methods related
to the data and datasets that we use for our project.

We define our LunarDataset and LunarDataLoader custom classes that are child
classes of torchvision.datasets.DatasetFolder and torch.utils.data.DataLoader
classes, respectively.
"""
from torchvision import transforms
from torchvision.datasets.folder import DatasetFolder, default_loader
from torch.utils.data import DataLoader


class LunarDataset(DatasetFolder):
    """A custom DatasetFolder class. Subclasses from
    torchvision.datasets.DatasetFolder."""

    def __init__(self, root=None, mode=None, verbose=True):
        self.root = root
        self.mode = mode
        self.verbose = verbose
        self.img_extensions = (".jpg", ".jpeg", ".tif", ".tiff")

        if mode == "train":
            self.transform = transforms.Compose(
                [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
            )
        elif mode == "val":
            self.transform = transforms.Compose(
                [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
            )
        elif mode == "test":
            self.transform = transforms.Compose(
                [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
            )

        super(LunarDataset, self).__init__(
            root,
            loader=default_loader,
            extensions=self.img_extensions,
            transform=self.transform,
        )

        if self.verbose:
            print(f"Image folder of mode {self.mode} \n {self}")


class LunarDataLoader(DataLoader):
    """A custom DataLoader class for our lunar anomalies project.
    Subclasses from torch.utils.data.DataLoader.
    """

    def __init__(
        self,
        dataset,
        sampler=None,
        batch_size=None,
        shuffle=None,
        num_workers=8,
        pin_memory=True,
        **kwargs,
    ):
        self.sampler = sampler
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.kwargs = kwargs
        super(LunarDataLoader, self).__init__(
            dataset,
            sampler=self.sampler,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            **kwargs,
        )


def get_scores_array(df, crop_size=64):
    """Function to get the anomaly_score plot in spatial coherence to the
    original image.

    Args:
        df: Dataframe with the sample_filepaths, y_true labels and
        anomaly_scores for the crops.
        crop_size: Size of the cropped image.

    Returns:
        y_true_placeholder: 2D numpy array for the Y_true labels in spatial
        coherence with the complete LROC image.
        anomaly_score_placeholder: 2D numpy array for the anomaly_scores in
        spatial coherence with the complete LROC image.
        coordinates: Coordinates tuple of the extreme points of the complete
        LROC image [X_min, Y_min, X_max, Y_max].
    """
    filepaths = df["sample_filepath"]
    anomaly_scores = df["anomaly_score"]
    y_true = df["y_true"]

    # Parsing the sample_filepaths strings to get the x, y pixel locations.
    x, y = [], []
    for paths in filepaths:
        x.append(int((((paths.split("/")[-1]).split(".")[0]).split("_")[1])[1:]))
        y.append(int((((paths.split("/")[-1]).split(".")[0]).split("_")[2])[1:]))

    # Getting the max and min coordinates of the extreme pixel locations of the
    # complete LROC image. Adding the crop_size to the X_max and Y_max to get
    # the actual maximum coordinate of the real LROC image.
    x_max, x_min = max(x), min(x)
    y_max, y_min = max(y), min(y)
    x_max += crop_size
    y_max += crop_size
    coordinates = (x_min, y_min, x_max, y_max)

    # Placeholder for reconstructing the global image map.
    rows, cols = (x_max - x_min), (y_max - y_min)
    anomaly_placeholder = np.zeros(shape=(rows, cols))
    y_true_placeholder = np.zeros(shape=(rows, cols))

    # Populating the anomaly_scores and y_true labels in the required locations
    # in the 2D array.
    for i in range(len(y_true)):
        x, y = x[i] - x_min, y[i] - y_min
        anomaly_placeholder[x : x + crop_size, y : y + crop_size] = anomaly_scores[i]
        y_true_placeholder[x : x + crop_size, y : y + crop_size] = y_true[i]

    return y_true_placeholder, anomaly_placeholder, coordinates
