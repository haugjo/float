"""Standard Scatter Plot.

This function returns a scatter plot using the style and coloring of the float framework.

Copyright (C) 2022 Johannes Haug.
"""
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from typing import List

# Global color palette
# dark blue, teal, light blue, dark green, olive, yellow green, red, magenta, grey, black
_PALETTE = ['#003366', '#44aa99', '#88ccee', '#117733', '#999933', '#ddcc77', '#cc3311', '#ee3377', '#bbbbbb', '#000000']


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
