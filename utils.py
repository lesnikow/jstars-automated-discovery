# -*- coding: utf-8 -*-
"""utils module
Define our utils module that encapsulates utility attributes and methods
 we use for our lunar anomalies project.
"""
import os
import time


class Utilitator:
    """A utilities class used for making experiment directory strings
    and experiment subdirectories, among other things.
    """

    def __init__(self, verbose: bool = True, mode: str = None):
        self.verbose = verbose
        self.mode = mode
        if self.mode == "" or not self.mode:
            raise ValueError(
                f"Please provide a Utilitator class mode in ['val', 'test', 'train']"
            )
        super(Utilitator).__init__()

    def return_experiment_directory_string(self, experiment_uid=None):
        # time_zero is the time on beginning of UNIX time.
        time_zero = 0
        experiment_directory_stamp = "exp_time_{}".format(int(time.time()) - time_zero)
        repository_home_directory = "/home/adam/lunar-anomalies/"
        experiment_directory = os.path.join(
            repository_home_directory,
            "results",
            (experiment_directory_stamp + "__" + experiment_uid),
        )
        if self.verbose is True:
            print(
                "The experiment directory where our experiment files will be"
                f"stored is: {experiment_directory}."
            )
        return experiment_directory

    def make_subdirectories_from_array(self, experiment_directory, sub_dirs: [str]):
        """Makes a list of subdirs given a list of subdir names."""
        for sub_dir in sub_dirs:
            try:
                if self.verbose:
                    print(
                        f"experiment_directory, sub_dir are {experiment_directory} {sub_dir}"
                    )
                os.makedirs(os.path.join(experiment_directory, sub_dir))
            except FileExistsError:
                print(f"{sub_dir} directory already exists.")

    def make_directories_from_experiment_directory(
        self, experiment_directory: str = None
    ):
        """Makes directories for a given experiment directory.

        Args:
            experiment_directory: experiment directory to make and where
                to create all experiment directory subdirectories. Typically
                this is genereated by our return_experiment_directory_string()
                class method.

        Returns: None
        """

        sub_dirs_all_modes = ["notebooks", "curves"]
        sub_dirs_val_test = ["distribution_plots", "most_anomalous_samples"]
        sub_dirs_train = [
            "models",
            "loss_graphs",
            "loss_tables",
            "reconstructions_and_generated_samples/reconstructions",
            "reconstructions_and_generated_samples/generated_samples",
        ]

        self.make_subdirectories_from_array(experiment_directory, sub_dirs_all_modes)
        if self.mode in ["val", "test"]:
            self.make_subdirectories_from_array(experiment_directory, sub_dirs_val_test)
        if self.mode in ["train"]:
            self.make_subdirectories_from_array(experiment_directory, sub_dirs_train)

        if self.verbose:
            print(f"{Utilitator.__name__} class says: I'm finished making directories.")
