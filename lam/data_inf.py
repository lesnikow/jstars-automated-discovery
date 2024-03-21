# -*- coding: utf-8 -*-
"""data_inf module
"""
from torchvision import transforms
from torchvision.datasets.folder import DatasetFolder, default_loader
from torch.utils.data import DataLoader


class InfDataset(DatasetFolder):
    """A custom DatasetFolder class.
    Subclasses from torchvision.datasets.DatasetFolder.
    """

    def __init__(self, root=None, mode="inf", verbose=True):
        self.root = root
        self.mode = mode
        self.verbose = verbose
        self.img_extensions = (".tif", ".tiff", ".jpg", ".jpeg")

        if mode == "inf":
            self.transform = transforms.Compose(
                [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
            )
        super(InfDataset, self).__init__(
            root,
            loader=default_loader,
            extensions=self.img_extensions,
            transform=self.transform,
        )


class InfDataLoader(DataLoader):
    """A custom DataLoader class for GCP inferencing.
    Subclasses from torch.utils.data.DataLoader.
    """

    def __init__(
        self,
        dataset,
        sampler=None,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        **kwargs,
    ):
        self.sampler = sampler
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.kwargs = kwargs
        super(InfDataLoader, self).__init__(
            dataset,
            sampler=self.sampler,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=num_workers,
            **kwargs,
        )
