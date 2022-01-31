from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
import warnings

# Global color palette
# dark blue, teal, light blue, dark green, olive, yellow green, red, magenta, grey, black
_PALETTE = ['#003366', '#44aa99', '#88ccee', '#117733', '#999933', '#ddcc77', '#cc3311', '#ee3377', '#bbbbbb', '#000000']


def spider_chart(measures: List[List],
                 metric_names: List[str],
                 legend_names: List[str],
                 ranges: Optional[List[Tuple]] = None,
                 invert: Optional[List[bool]] = None) -> Axes:
    """Returns a spider chart that shows the specified metric values.

    Args:
        measures: A list of lists, where each list corresponds to a series of measurements.
        metric_names: The name of the performance metrics that are displayed.
        legend_names: Labels for each list of measurements. These labels will be used in the legend.
        ranges:
            The ranges for each of the metrics. If None, will be set to (0,1) for each metric. If not None, each tuple
            in the list corresponds to a metric and will be set to (0,1) if None.
        invert:
            A list of bool values corresponding to if the metric should be inverted or not. If None, will be set to False
            for each value.

    Returns:
        Axes: The Axes object containing the plot.
    """
    if len(measures) > 3:
        warnings.warn('Plotting more than three measures in a spider chart can make it difficult to read.')

    if len(metric_names) > 8:
        warnings.warn('Plotting more than eight variables in a spider chart can make it difficult to read.')

    # Set up ranges for each metric
    invert = invert if invert else [False for _ in range(len(metric_names))]
    ranges = [r if r else (0, 1) for r in ranges] if ranges else [(0, 1) for _ in range(len(metric_names))]
    ranges = [(r2, r1) if i else (r1, r2) for (r1, r2), i in zip(ranges, invert)]

    # Set up figure and axes for each metric
    fig = plt.figure()
    angles = np.arange(0, 360, 360. / len(metric_names))
    axes = [fig.add_axes([0.1, 0.1, 1.06, 0.81], polar=True, label="axes{}".format(i)) for i in range(len(metric_names))]

    # Draw metric names
    _, text = axes[0].set_thetagrids(angles, labels=metric_names)
    labels = []
    for label, angle in zip(text, angles):
        x, y = label.get_position()
        lab = axes[0].text(x, y - 0.03, label.get_text(), transform=label.get_transform(), ha=label.get_ha(), va=label.get_va())
        lab.set_rotation(angle - 90) if angle < 180 else lab.set_rotation(angle + 90)
        labels.append(lab)
    axes[0].set_xticklabels([])

    # Set up grid
    for ax in axes[1:]:
        ax.patch.set_visible(False)
        ax.grid("off")
        ax.xaxis.set_visible(False)
    for i, ax in enumerate(axes):
        grid = np.linspace(*ranges[i], num=6)
        grid_label = ["{}".format(round(x, 2)) for x in grid]
        grid_label[0] = ""
        ax.set_rgrids(grid, labels=grid_label, angle=angles[i])
        ax.set_ylim(*ranges[i])

    # Plot the measures
    angle = np.deg2rad(np.r_[angles, angles[0]])
    ax = axes[0]
    for i in range(len(measures)):
        data_scaled = _scale_data(measures[i], ranges)
        ax.plot(angle, np.r_[data_scaled, data_scaled[0]], linewidth=2, color=_PALETTE[i], label=legend_names[i])
        ax.fill(angle, np.r_[data_scaled, data_scaled[0]], color=_PALETTE[i], alpha=0.2)
        ax.legend(loc='lower left', bbox_to_anchor=(-0.54, -0.12), frameon=False)

    return ax


def _scale_data(measures: List[float],
                ranges: List[Tuple]) -> List[float]:
    """Scales measures to specified range and inverts them if the range is reversed.

    Args:
        measures: A list containing a series of measurements.
        ranges: The ranges for each of the metrics.

    Returns:
        List[float]: the rescaled measures
    """
    # Rescale the first measure and save its range
    range_min_init, range_max_init = ranges[0]
    measure = measures[0]
    if range_min_init > range_max_init:
        # Update measure if range is inverted
        measure = range_max_init - (measure - range_min_init)
        range_min_init, range_max_init = range_max_init, range_min_init
    measures_scaled = [measure]

    # Rescale the remaining measures w.r.t. to their own range and the range of the first measure
    for measure, (range_min, range_max) in zip(measures[1:], ranges[1:]):
        if range_min > range_max:
            # Update measure if range is inverted
            measure = range_max - (measure - range_min)
            range_min, range_max = range_max, range_min
        measures_scaled.append((measure - range_min) / (range_max - range_min) *
                               (range_max_init - range_min_init) + range_min_init)
    return measures_scaled
