"""Feature Selection Scatter Plot.

This function returns a special scatter plot that illustrates the selected features over time of one or multiple online
feature selection methods.

Copyright (C) 2022 Johannes Haug.
"""
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from typing import List

# Global color palette
# dark blue, teal, light blue, dark green, olive, yellow green, red, magenta, grey, black
_PALETTE = ['#003366', '#44aa99', '#88ccee', '#117733', '#999933', '#ddcc77', '#cc3311', '#ee3377', '#bbbbbb', '#000000']


def feature_selection_scatter(selected_features: List[list],
                              fig_size: tuple = (13, 5),
                              font_size: int = 16) -> Axes:
    """Returns a scatter plot that illustrates the selected features over time for the specified models.

    Args:
        selected_features:
            A list of lists, where each list corresponds to the selected feature vectors of one feature selector.
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
