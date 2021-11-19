from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from typing import List, Optional

# Global color palette
# dark blue, teal, light blue, dark green, olive, yellow green, red, magenta, grey, black
_PALETTE = ['#003366', '#44aa99', '#88ccee', '#117733', '#999933', '#ddcc77', '#cc3311', '#ee3377', '#bbbbbb', '#000000']


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
