from matplotlib.axes import Axes
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union

# Global color palette
# dark blue, light blue, teal, dark green, olive, yellow green, red, magenta, grey, black
_PALETTE = ['#003366', '#88ccee', '#44aa99', '#117733', '#999933', '#ddcc77', '#cc3311', '#ee3377', '#bbbbbb', '#000000']


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
