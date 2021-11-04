"""Visualization Module.

This module contains visualizations that may be used to illustrate the test results of online predictive models,
online feature selection methods and concept drift detection methods. We recommend combining these visualizations with
the float evaluator and pipeline modules to deliver high-quality and standardized experiments.

Copyright (C) 2021 Johannes Haug

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from matplotlib.axes import Axes
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from typing import List, Optional, Union

# Global color palette
# dark blue, light blue, teal, dark green, olive, yellow green, red, magenta, grey, black
_PALETTE = ['#003366', '#88ccee', '#44aa99', '#117733', '#999933', '#ddcc77', '#cc3311', '#ee3377', '#bbbbbb', '#000000']


def plot(measures: List[list],
         legend_labels: List[str],
         y_label: str,
         fig_size: tuple = (13, 5),
         font_size: int = 16,
         x_label: str = 'Time Step $t$',
         variance_measures: Optional[List[list]] = None,
         apply_smoothing: bool = False) -> Axes:
    """Returns a line plot.

    Each list provided in the measures attribute is displayed as one line.

    Args:
        measures: A list of lists, where each list corresponds to a series of measurements.
        legend_labels: Labels for each list of measurements. These labels will be used in the legend.
        y_label: The y-axis label text (e.g. the name of the performance measure that is displayed).
        fig_size: The figure size (length x height)
        font_size: The font size of the axis labels.
        x_label: The x-axis label text. This defaults to 'Time Step t'.
        variance_measures:
            Optionally, one can depict variances (as shaded areas around the line plot). This parameter must have the
            same dimensionality as 'measures'.
        apply_smoothing:
            If true, we apply a savgol_filter to the provided measures. However, note that this may distort the actual
            results or hide interesting effects. In general, we do not recommend to apply a smoothing.

    Returns:
        Axes: The Axes object containing the line plot.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_axisbelow(True)
    ax.grid(True)

    for i in range(len(measures)):
        if apply_smoothing:  # Apply a filter to the provided measurements to smooth the line plots
            y = savgol_filter(measures[i], 51, 3)
        else:
            y = measures[i]

        ax.plot(np.arange(len(measures[i])), y, color=_PALETTE[i], label=legend_labels[i], lw=2)

        if variance_measures:  # Display variances as shaded areas around the line plot
            ax.fill_between(np.arange(len(measures[i])),
                            y - np.array(variance_measures[i]),
                            y + np.array(variance_measures[i]),
                            color=_PALETTE[i],
                            alpha=0.5)

    ax.set_xlabel(x_label, size=font_size, labelpad=1.6)
    ax.set_ylabel(y_label, size=font_size, labelpad=1.6)
    plt.legend(frameon=True, loc='best', fontsize=font_size * 0.8, borderpad=0.3, handletextpad=0.5)
    plt.margins(0.01, 0.03)
    plt.tick_params(axis='both', labelsize=font_size * 0.8, length=0)
    plt.tight_layout()
    return ax


def scatter(measures: List[list],
            legend_labels: List[str],
            y_label: str,
            fig_size: tuple = (13, 5),
            font_size: int = 16,
            x_label: str = 'Time Step $t$') -> Axes:
    """Returns a scatter plot.

    Each list provided in the measures attribute is displayed in a different color.

    Args:
        measures: A list of lists, where each list corresponds to a series of measurements.
        legend_labels: Labels for each list of measurements. These labels will be used in the legend.
        y_label: The y-axis label text (e.g. the name of the performance measure that is displayed).
        fig_size: The figure size (length x height)
        font_size: The font size of the axis labels.
        x_label: The x-axis label text. This defaults to 'Time Step t'.

    Returns:
        Axes: The Axes object containing the scatter plot.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_axisbelow(True)
    ax.grid(True)

    for i in range(len(measures)):
        ax.scatter(np.arange(len(measures[i])),
                   measures[i],
                   color=_PALETTE[i],
                   label=legend_labels[i])

    ax.set_xlabel(x_label, size=font_size, labelpad=1.6)
    ax.set_ylabel(y_label, size=font_size, labelpad=1.6)
    plt.legend(frameon=True, loc='best', fontsize=font_size * 0.8, borderpad=0.3, handletextpad=0.5)
    plt.margins(0.01, 0.03)
    plt.tick_params(axis='both', labelsize=font_size * 0.8, length=0)
    plt.tight_layout()
    return ax


def bar(measures: List[list],
        legend_labels: List[str],
        y_label: str,
        fig_size: tuple = (13, 5),
        font_size: int = 16,
        x_label: str = 'Time Step $t$') -> Axes:
    """Returns a bar plot.

    Args:
        measures: A list of lists, where each list corresponds to a series of measurements.
        legend_labels: Labels for each list of measurements. These labels will be used in the legend.
        y_label: The y-axis label text (e.g. the name of the performance measure that is displayed).
        fig_size: The figure size (length x height)
        font_size: The font size of the axis labels.
        x_label: The x-axis label text. This defaults to 'Time Step t'.

    Returns:
        Axes: The Axes object containing the bar plot.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y')
    width = 0.8
    n_measures = len(measures)

    for i in range(n_measures):
        ax.bar(np.arange(n_measures) - width / 2. + i / n_measures * width,
               measures[i],
               width=width / n_measures,
               align="edge",
               color=_PALETTE[i],
               label=legend_labels[i])

    ax.set_xlabel(x_label, size=font_size, labelpad=1.6)
    ax.set_ylabel(y_label, size=font_size, labelpad=1.6)
    plt.legend(frameon=True, loc='best', fontsize=font_size * 0.8, borderpad=0.3, handletextpad=0.5)
    plt.margins(0.01, 0.03)
    plt.tick_params(axis='both', labelsize=font_size * 0.8, length=0)
    plt.tight_layout()
    return ax


def feature_selection_scatter(selected_features: list,
                              fig_size: tuple = (13, 5),
                              font_size: int = 16) -> Axes:
    """Return a scatter plot that illustrates the selected features over time.

    Args:
        selected_features:
            A list corresponding to the selected feature vectors of a feature selection model.
        fig_size: The figure size (length x height)
        font_size: The font size of the axis labels.

    Returns:
        Axes: The Axes object containing the plot.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_axisbelow(True)
    ax.grid(True)

    x, y = [], []
    for k, val in enumerate(selected_features):
        x.extend(np.ones(len(val), dtype=int) * k)
        y.extend(val)

    ax.set_xlabel('Time Step $t$', size=font_size, labelpad=1.6)
    ax.set_ylabel('Feature Index', size=font_size, labelpad=1.5)
    ax.scatter(x, y, marker='.', zorder=100, color=_PALETTE[0], label='Selected Feature Indicator')
    ax.legend(frameon=True, loc='best', fontsize=font_size * 0.8, borderpad=0.2, handletextpad=0.2)

    plt.margins(0.01, 0.03)
    plt.tick_params(axis='both', labelsize=font_size * 0.8, length=0)
    plt.tight_layout()
    return ax


def feature_selection_bar(selected_features: List[list],
                          model_names: List[str],
                          feature_names: list,
                          top_n_features: Optional[int] = None,
                          fig_size: tuple = (13, 5),
                          font_size: int = 16) -> Axes:
    """Returns a bar plot that shows the number of times a feature was selected (between multiple models).

    Args:
        selected_features:
            A list of lists, where each list corresponds the selected feature vectors of one feature selection model.
        model_names: Names of the feature selection models. These labels will be used in the legend.
        feature_names: The names of all input features. The feature names will be used as x-tick labels.
        top_n_features:
            Specifies the top number of features to be displayed. If the attribute is None, we show all features in
            their original order. If the attribute is not None, we select the top features of the first provided model
            and compare it with the remaining models.
        fig_size: The figure size (length x height)
        font_size: The font size of the axis labels.

    Returns:
        Axes: The Axes object containing the bar plot.
    """
    width = 0.8
    n_models = len(selected_features)
    n_features = len(feature_names) if top_n_features is None else top_n_features

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y')
    order = None
    name_idx = np.arange(n_features)

    for i in range(n_models):
        meas = np.array(selected_features[i]).flatten()
        uniques, counts = np.unique(meas, return_counts=True)

        if top_n_features is not None:
            if order is None:  # We order the features according to the first provided model.
                order = np.argsort(counts)[::-1][:top_n_features]
                name_idx = uniques[order]
            y = counts[order]
        else:
            y = np.zeros(n_features)
            y[uniques] = counts

        ax.bar(np.arange(n_features) - width / 2. + i / n_models * width,
               y,
               width=width / n_models,
               zorder=100,
               color=_PALETTE[i],
               label=model_names[i])

    plt.xticks(np.arange(n_features),
               labels=np.asarray(feature_names)[name_idx],
               rotation=20,
               ha='right')
    plt.ylabel('No. Times Selected', size=font_size, labelpad=1.5)
    plt.xlabel('Input Feature', size=font_size, labelpad=1.6)
    plt.legend(frameon=True, loc='best', fontsize=font_size * 0.8, borderpad=0.3, handletextpad=0.5)
    plt.margins(0.01, 0.03)
    plt.tick_params(axis='both', labelsize=font_size * 0.8, length=0)
    plt.tight_layout()
    return ax


def concept_drift_detection_scatter(detected_drifts: List[list],
                                    model_names: List[str],
                                    n_samples: int,
                                    known_drifts: Union[List[int], List[tuple]],
                                    batch_size: int,
                                    n_pretrain: int,
                                    fig_size: tuple = (13, 5),
                                    font_size: int = 16) -> Axes:
    """Returns a scatter plot with the known and the detected concept drifts.

    Args:
        detected_drifts:
            A list of lists, where each list corresponds the detected drifts of one concept drift detector.
        model_names: Names of the concept drift detection models. These labels will be used in the legend.
        n_samples: The total number of samples observed.
        known_drifts (List[int] | List[tuple]):
            The positions in the dataset (indices) corresponding to known concept drifts.
        batch_size:
            The batch size used for the evaluation of the data stream. This is needed to translate the known drift
            positions to logical time steps (which is the format of the detected drifts).
        n_pretrain:
            The number of observations used for pre-training. This number needs to be subtracted from the known drift
            positions in order to translate them to the correct logical time steps.
        fig_size: The figure size (length x height)
        font_size: The font size of the axis labels.

    Returns:
        Axes: The Axes object containing the plot.
    """
    n_models = len(detected_drifts)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_axisbelow(True)
    ax.grid(True, axis='x')

    # Draw known drifts
    for known_drift in known_drifts:
        if isinstance(known_drift, tuple):
            ax.axvspan(round((known_drift[0] - n_pretrain) / batch_size),
                       round((known_drift[1] - n_pretrain) / batch_size),
                       color=_PALETTE[-3],
                       alpha=0.5,
                       hatch="//")
        else:
            ax.axvline(round((known_drift - n_pretrain) / batch_size), color=_PALETTE[-3], lw=3, zorder=0)

    # Draw detected drifts
    y_loc = 0
    y_tick_labels = []
    for i in range(n_models):
        ax.axhline(y_loc, color=_PALETTE[0], zorder=5)
        ax.scatter(detected_drifts[i],
                   np.repeat(y_loc, len(detected_drifts[i])),
                   marker='|',
                   color=_PALETTE[0],
                   s=500,
                   zorder=10)
        y_loc += (1 / n_models)
        y_tick_labels.append(model_names[i])

    plt.yticks(np.arange(0, 1, 1 / n_models), y_tick_labels)
    plt.xticks(np.arange(0, ((n_samples - n_pretrain) / batch_size) - 10,
                         round(((n_samples - n_pretrain) / batch_size) * 0.1)))
    plt.xlim(-((n_samples - n_pretrain) / batch_size) * 0.005,
             ((n_samples - n_pretrain) / batch_size) + ((n_samples - n_pretrain) / batch_size) * 0.005)
    plt.xlabel('Time Step $t$', fontsize=font_size)

    known_drift_patch = mlines.Line2D([], [], marker='|', linestyle='None', markersize=10, markeredgewidth=2,
                                      color=_PALETTE[-3], label='Known drifts')
    detected_drift_patch = mlines.Line2D([], [], marker='|', linestyle='None', markersize=10, markeredgewidth=2,
                                         color=_PALETTE[0], label='Detected drifts')

    plt.legend(handles=[known_drift_patch, detected_drift_patch],
               frameon=True,
               loc='best',
               fontsize=font_size * 0.8,
               borderpad=0.3,
               handletextpad=0.5)

    plt.margins(0.01, 0.1)
    plt.tick_params(axis='both', labelsize=font_size * 0.8, length=0)
    plt.tight_layout()
    return ax
