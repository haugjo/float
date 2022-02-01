from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional

# Global color palette
# dark blue, teal, light blue, dark green, olive, yellow green, red, magenta, grey, black
_PALETTE = ['#003366', '#44aa99', '#88ccee', '#117733', '#999933', '#ddcc77', '#cc3311', '#ee3377', '#bbbbbb', '#000000']


def spider_chart(measures: List[List],
                 measure_names: List[str],
                 legend_names: List[str],
                 ranges: Optional[List[Tuple]] = None,
                 invert: Optional[List[bool]] = None,
                 fig_size: tuple = (8, 5),
                 font_size: int = 16) -> Axes:
    """Returns a spider chart that shows the specified metric values.

    Args:
        measures:
            A list of lists. Each list corresponds to different (e.g. model-wise) results for one performance measure.
        measure_names:
            The names of the performance measures that are shown. This attribute has the same length as 'measures'
        legend_names:
            Legend labels for each different result per measure (e.g. model names). This attribute has the same length
            as each list in the 'measures' attribute.
        ranges:
            The value ranges for each of the measures. If None, the range will be set to (0,1) per default. Otherwise,
            each tuple in the list corresponds to the range of the measure at the respective position in 'measures'.
        invert:
            A list of bool values indicating for each measure if it should be inverted. We invert a measure if a
            lower value is better than higher value. Otherwise, the spider chart may be confusing. If None, 'invert'
            will be set to False for each measure.
        fig_size: The figure size (length x height).
        font_size: The font size of the axis labels.

    Returns:
        Axes: The Axes object containing the plot.
    """
    # Set default 'invert' and 'ranges' where no value has been specified.
    if invert is None:
        invert = [False for _ in range(len(measure_names))]  # No measure will be inverted.
    if ranges is None:
        ranges = [(0, 1) for _ in range(len(measure_names))]  # Use the default range for each measure.
    else:
        ranges = [rg if rg is not None else (0, 1) for rg in ranges]  # Use the default, where a value is unspecified.

    # Compute the inverted ranges, as specified.
    inverted_ranges = []
    for rg, inv in zip(ranges, invert):
        if inv:
            inverted_ranges.append((rg[1], rg[0]))
        else:
            inverted_ranges.append(rg)
    ranges = inverted_ranges

    # Reformat and scale all measures.
    scaled_measures = []
    for meas, rg in zip(measures, ranges):
        # Rescale the measures
        inverted = False
        if rg[0] > rg[1]:
            inverted = True
            rg_min = rg[1]
            rg_max = rg[0]
        else:
            rg_min = rg[0]
            rg_max = rg[1]

        meas = np.asarray(meas)
        meas = ((meas - rg_min) / (rg_max - rg_min)) * (1 - 0) + 0

        for i, model in enumerate(meas):
            if len(scaled_measures) < len(meas):
                if inverted:
                    scaled_measures.append([1 - model])
                else:
                    scaled_measures.append([model])
            else:
                if inverted:
                    scaled_measures[i].append(1 - model)
                else:
                    scaled_measures[i].append(model)

    # Set up the figure and axes for each metric.
    fig = plt.figure(figsize=fig_size)
    angles = np.arange(0, 360, 360. / len(measure_names))
    axes = [fig.add_axes([0.15, 0.09, 1.06, 0.81], polar=True, label="axes{}".format(i)) for i in range(len(measure_names))]

    # Draw the measure names.
    _, text = axes[0].set_thetagrids(angles, labels=measure_names)
    labels = []
    for label, angle in zip(text, angles):
        x, y = label.get_position()
        lab = axes[0].text(x,
                           y - 0.03,
                           label.get_text(),
                           transform=label.get_transform(),
                           ha=label.get_ha(),
                           va=label.get_va(),
                           fontsize=font_size)
        if angle < 180:
            lab.set_rotation(angle - 90)
        else:
            lab.set_rotation(angle + 90)
        labels.append(lab)
    axes[0].set_xticklabels([])

    # Draw the grid.
    for ax in axes[1:]:
        ax.patch.set_visible(False)
        ax.grid("off")
        ax.xaxis.set_visible(False)

    for i, ax in enumerate(axes):
        grid = np.linspace(0, 1, num=6)
        grid_label = ["{}".format(round(x, 2)) for x in np.linspace(*ranges[i], num=6)]
        grid_label[0] = ""
        grid_label[-1] = ""
        ax.set_rgrids(grid, labels=grid_label, angle=angles[i], fontsize=font_size * 0.8)
        ax.set_ylim((0, 1))

    # Plot the scaled measures.
    angle = np.deg2rad(np.r_[angles, angles[0]])
    ax = axes[0]
    for i, model_meas in enumerate(scaled_measures):
        ax.plot(angle, np.r_[model_meas, model_meas[0]], linewidth=2, color=_PALETTE[i], label=legend_names[i])
        ax.fill(angle, np.r_[model_meas, model_meas[0]], color=_PALETTE[i], alpha=0.2)
        ax.legend(loc='lower left', bbox_to_anchor=(-0.82, -0.1), frameon=True, prop={'size': font_size * 0.8})

    return ax
