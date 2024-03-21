# -*- coding: utf-8 -*-
"""metrics module
Define our metrics module that encapsulates attributes and methods related to
the metrics we use for our lunar anomalies project.
"""
import matplotlib.pyplot as plt
import numpy as np
from inspect import signature
from sklearn.metrics import average_precision_score, auc
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import PrecisionRecallDisplay


class Curve:
    def __init__(
        self,
        y_test: [] = None,
        y_scores: [] = None,
        pos_label: int = None,
        feature_str: str = None,
        save_to_disk_fp: str = None,
    ):
        """Init method for our Curve class. This is intended to be the parent
        class for children classes like PrecisionRecallCurve and
        RecieverOperatorCharacteristicCurve.

        Args:
            y_test: the list of true labels
            y_scores: the list of model scores
            pos_label: the torch index to use as our positive class
            save_to_disk_fp: the string where to save the plot

        Returns:
            None

        """
        self.y_test = y_test
        self.y_scores = y_scores
        self.pos_label = pos_label
        self.feature_str = feature_str
        self.save_to_disk_fp = save_to_disk_fp

    def make_plot(
        self,
        show: bool = None,
        verbose: bool = False,
        plot_type: str = None,
    ):
        """Makes a Curve plot. This is intended to be overriden by Curve
        children classes like PrecisionRecallCurve or
        RecieverOperatingCharacteristicCurve.

        Args:
            show: Whether to show the plot on-scren via e.g. plt.show()
            verbose: Whether to enable verbose mode to print various
                information on screen.
            plot_type: String of plot type to use. Possible choices include
                "pr" and "roc".

        Returns:
           None
        """

        xlarge_fontsize = 24
        large_fontsize = 20
        medium_fontsize = 18

        if plot_type == "pr":
            self.x_axis_values = self.recalls
            self.y_axis_values = self.precisions
            self.x_axis_label = "Recall"
            self.y_axis_label = "Precision"
            self.plot_title = self.feature_str.title()

        elif plot_type == "roc":
            self.x_axis_values = self.fpr
            self.y_axis_values = self.tpr
            self.x_axis_label = "False Positive Rate"
            self.y_axis_label = "True Positive Rate"
            self.plot_title = f"ROC Curve. AUC={self.roc_auc:0.2f}"

        # Set plot title, labels.
        plt.title(self.plot_title, fontsize=xlarge_fontsize, fontweight="bold")
        plt.xlabel(self.x_axis_label, fontsize=large_fontsize, fontweight="bold")
        plt.ylabel(self.y_axis_label, fontsize=large_fontsize, fontweight="bold")

        # Set axes limits.
        plt.xlim([0.0, 1.0])
        if plot_type == "pr":
            plt.ylim([0.0, 1.05 * max(self.y_axis_values[:-2])])
        elif plot_type == "roc":
            plt.ylim([0.0, 1.05])
        # Hack for s3 ylim.
        if self.feature_str == "s3":
            plt.ylim([0.0, 5.0e-5])

        # Plot the curve.
        plt.plot(
            self.x_axis_values,
            self.y_axis_values,
            label="Model",
        )

        # Fill the area under the curve.
        plt.fill_between(
            self.x_axis_values,
            self.y_axis_values,
            alpha=0.2,
            color="b",
            interpolate=False,
        )

        # Horizontal line at the random baseline, if plot_type is "pr".
        if plot_type == "pr":
            plt.axhline(
                y=self.random_precision,
                color="#FF0000",
                linestyle="--",
                label="Random baseline",
            )

        # Show the AP versus random baseline improvement factor.
        plt.text(
            0.05,
            0.25,
            f"Improvement factor = {self.improvement_factor:0.2f}",
            fontsize=medium_fontsize,
            fontweight="bold",
            transform=plt.gca().transAxes,
        )

        # Show the AP.
        plt.text(
            0.05,
            0.15,
            f"Model average precision = {self.average_precision:0.2e}",
            fontsize=medium_fontsize,
            fontweight="normal",
            transform=plt.gca().transAxes,
        )

        # Show the random baseline.
        plt.text(
            0.05,
            0.05,
            f"Random baseline = {self.random_precision:0.2e}",
            fontsize=medium_fontsize,
            fontweight="normal",
            transform=plt.gca().transAxes,
        )

        # Make x, y axis labels, ticks bigger.
        plt.xlabel(self.x_axis_label, fontsize=large_fontsize, fontweight="bold")
        plt.ylabel(self.y_axis_label, fontsize=large_fontsize, fontweight="bold")
        plt.xticks(fontsize=large_fontsize)
        plt.yticks(fontsize=large_fontsize)

        # Make y axis have scientific notation.
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        # Plot the legend, setting the order of the legend.
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0, 1]
        plt.legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc="upper right",
            fontsize=medium_fontsize,
        )

        # Save, show, close.
        if self.save_to_disk_fp is not None:
            plt.savefig(
                self.save_to_disk_fp,
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        plt.close()


class RecieverOperatingCharacteristicCurve(Curve):
    def __init__(
        self,
        y_test: [] = None,
        y_scores: [] = None,
        pos_label: int = None,
        drop_intermediate: bool = True,
        save_to_disk_fp: str = None,
    ):
        """Init method for the RecieverOperatingCharacteristicCurve class.
        Args:
            y_test: the list of true labels
            y_scores: the list of model scores
            pos_label: the torch index to use as our positive class
            save_to_disk_fp: the string where to save the plot

        Returns:
            None
        """
        super().__init__(
            y_test, y_scores, pos_label=pos_label, save_to_disk_fp=save_to_disk_fp
        )
        self.drop_intermediate = drop_intermediate
        self.fpr, self.tpr, self.thresholds = roc_curve(
            self.y_test,
            self.y_scores,
            pos_label=self.pos_label,
            drop_intermediate=self.drop_intermediate,
        )
        self.roc_auc = auc(self.fpr, self.tpr)

    def make_plot(
        self,
        show: bool = None,
        verbose: bool = False,
        plot_type: str = "roc",
    ):
        """Makes a reciever operating characteristic plot. This overrides Curve's
        implementaiton of this method.

        Args:
            show: Whether to show the plot on-scren via e.g. plt.show()
            verbose: Whether to enable verbose mode to print various
                information on screen.
            plot_type: String of plot type to use. Possible choices include
                "pr" and "roc".

        Returns:
           None
        """
        super().make_plot(show=show, verbose=verbose, plot_type=plot_type)


class PrecisionRecallCurve(Curve):
    def __init__(
        self,
        y_test: [] = None,
        y_scores: [] = None,
        pos_label: int = None,
        feature_str: str = None,
        save_to_disk_fp: str = None,
    ):
        """Init method for our PrecisionRecallCurve class.

        Args:
            y_test: the list of true labels
            y_scores: the list of model scores
            pos_label: the torch index to use as our positive class
            save_to_disk_fp: the string where to save our PR curve

        Returns:
            None
        """
        super().__init__(
            y_test,
            y_scores,
            pos_label=pos_label,
            feature_str=feature_str,
            save_to_disk_fp=save_to_disk_fp,
        )

        self.precisions, self.recalls, self.thresholds = precision_recall_curve(
            y_test, y_scores, pos_label=self.pos_label
        )
        self.average_precision = average_precision_score(
            y_test, y_scores, pos_label=self.pos_label
        )
        self.dataset_size = len(y_scores)
        self.count_positives = np.count_nonzero(y_test)
        if pos_label == 0:
            self.count_positives = self.dataset_size - self.count_positives
        self.random_precision = float(self.count_positives / self.dataset_size)
        self.improvement_factor = float(self.average_precision / self.random_precision)

    def make_plot(
        self,
        show: bool = None,
        verbose: bool = False,
        plot_type: str = "pr",
    ):
        """Makes a precision recall curve plot. This overrides Curve's
        implementaiton of this method.

        Args:
            show: Whether to show the plot on-scren via e.g. plt.show()
            verbose: Whether to enable verbose mode to print various
                information on screen.
            plot_type: String of plot type to use. Possible choices include
                "pr" and "roc".

        Returns:
           None
        """
        super().make_plot(show=show, verbose=verbose, plot_type=plot_type)

    
class KDE_Plot():
    def __init__(
        self,
        y_test: [] = None,
        y_scores: [] = None,
        pos_label: int = None,
        feature_str: str = None,
        save_to_disk_fp: str = None,
    ):
        """Init method for our KDE_Plot class.

        Args:
            y_test: the list of true labels
            y_scores: the list of model scores
            pos_label: the torch index to use as our positive class
            save_to_disk_fp: the string where to save our PR curve

        Returns:
            None
        """
        self.y_test = y_test
        self.y_scores = y_scores
        self.pos_label = pos_label
        self.feature_str = feature_str
        self.save_to_disk_fp = save_to_disk_fp

    def make_plot(
        self,
        show: bool = None,
        verbose: bool = False,
    ):
        """Makes a KDE plot. 

        Args:
            show: Whether to show the plot on-scren via e.g. plt.show()
            verbose: Whether to enable verbose mode to print various
                information on screen.

        Returns:
           None
        """
        number_of_positive_samples = 4
        ax = sns.kdeplot(y_scores, shade=True, bw=0.05, legend=True)
        ax.set_title("Positive Samples (bottom) Over Background Distribution")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        rugplot_positive_samples = sns.rugplot(
            y_scores[:number_of_positive_samples], height=0.05
        )
        figure_positive_samples = rugplot_positive_samples.get_figure()
        if self.save_to_disk_fp is not None:
            figure_positive_samples.savefig(self.save_to_disk_fp)

