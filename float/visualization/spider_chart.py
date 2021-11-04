from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import warnings

_PALETTE = ['#003366', '#88ccee', '#44aa99', '#117733', '#999933', '#ddcc77', '#cc3311', '#ee3377', '#bbbbbb', '#000000']


# TODO refactor code to have more meaningful variables and be more understandable, add comments, make plot look good

def _scale_data(measures: List[float],
                ranges: List[Tuple]) -> List[float]:
    """
    Scale measures to specified range and invert them if the range is reversed.

    Args:
        measures: A list containing a series of measurements.
        ranges: The ranges for each of the metrics.

    Returns:
        List[float]: the rescaled measures
    """
    # TODO simplify if possible
    for d, (y1, y2) in zip(measures[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = measures[0]
    if x1 > x2:
        d = x2 - (d - x1)
        x1, x2 = x2, x1
    data_scaled = [d]
    for d, (y1, y2) in zip(measures[1:], ranges[1:]):
        if y1 > y2:
            d = y2 - (d - y1)
            y1, y2 = y2, y1
        data_scaled.append((d - y1) / (y2 - y1) * (x2 - x1) + x1)
    return data_scaled


def spider_chart(measures: List[List],
                 metric_names: List[str],
                 legend_names: List[str],
                 ranges: List[Tuple],
                 invert: List[bool]) -> Axes:
    """
    Returns a spider chart that shows the specified metric values.

    Args:
        measures: A list of lists, where each list corresponds to a series of measurements.
        metric_names: The name of the performance metrics that are displayed.
        legend_names: Labels for each list of measurements. These labels will be used in the legend.
        ranges: The ranges for each of the metrics.
        invert: A list of bool values corresponding to if the metric should be inverted.

    Returns:
        Axes: The Axes object containing the plot.

    """
    if len(measures) > 3:
        warnings.warn('Plotting more than three measures in a spider chart can make it difficult to read.')

    # TODO figure out how to handle ranges that have a different scale than [0,1]
    # ranges = []
    # for i in range(len(measures[0])):
    #     metric_max = np.max([val[i] for val in measures])
    #     ranges.append((metric_max + 0.1 * metric_max, 0)) if invert[i] else ranges.append((0, metric_max + 0.1 * metric_max))

    # add axes for each metric
    fig = plt.figure()
    angles = np.arange(0, 360, 360. / len(metric_names))
    axes = [fig.add_axes([0.1, 0.1, 0.9, 0.81], polar=True, label="axes{}".format(i)) for i in range(len(metric_names))]

    # plot metric names
    _, text = axes[0].set_thetagrids(angles, labels=metric_names)
    labels = []
    for label, angle in zip(text, angles):
        x, y = label.get_position()
        lab = axes[0].text(x, y - 0.03, label.get_text(), transform=label.get_transform(), ha=label.get_ha(), va=label.get_va())
        lab.set_rotation(angle - 90)
        labels.append(lab)
    axes[0].set_xticklabels([])

    # initialize grid
    for ax in axes[1:]:
        ax.patch.set_visible(False)
        ax.grid("off")
        ax.xaxis.set_visible(False)

    # set up grid
    for i, ax in enumerate(axes):
        grid = np.linspace(*ranges[i], num=6)
        grid_label = ["{}".format(round(x, 2)) for x in grid]
        grid_label[0] = ""
        ax.set_rgrids(grid, labels=grid_label, angle=angles[i])
        ax.set_ylim(*ranges[i])

    # plot the measures
    angle = np.deg2rad(np.r_[angles, angles[0]])
    ax = axes[0]
    for i in range(len(measures)):
        data_scaled = _scale_data(measures[i], ranges)
        ax.plot(angle, np.r_[data_scaled, data_scaled[0]], color=_PALETTE[i], label=legend_names[i])
        ax.fill(angle, np.r_[data_scaled, data_scaled[0]], color=_PALETTE[i], alpha=0.2)
        ax.legend(loc='lower left', bbox_to_anchor=(-0.41, -0.12))

    return ax
