import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import warnings
from scipy.signal import savgol_filter
from skmultiflow.data.data_stream import Stream
import matplotlib.lines as mlines
from math import pi

# dark blue, light blue, teal, dark green, olive, yellow green, red, magenta, grey, black
palette = ['#003366', '#88ccee', '#44aa99', '#117733', '#999933', '#ddcc77', '#cc3311', '#ee3377', '#bbbbbb', '#000000']

font_size = 12


def plot(measures, labels, measure_name, measure_type, variances=None, fig_size=(10.2, 5.2), smooth_curve=False):
    """
    Creates a line plot.

    Args:
        measures (list[list]): the list of lists of measures to be visualized
        labels (list[str]): the list of labels for the models
        measure_name (str): the measure to be plotted
        measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        variances (list[list]): the list of lists of the measures' variance values
        fig_size (float, float): the figure size of the plot
        smooth_curve (bool | list[bool]): True if the plotted curve should be smoothed, False otherwise

    Returns:
        Axes: the Axes object containing the line plot
    """
    if measure_type not in ['prediction', 'change_detection', 'feature_selection']:
        warnings.warn(f'Only measures of type "prediction", "change_detection" or "feature_selection" can be visualized with method plot.')
        return

    fig, ax = plt.subplots(figsize=fig_size)
    for i, (measure, label) in enumerate(zip(measures, labels)):
        smooth_curve_i = smooth_curve if type(smooth_curve) is bool else smooth_curve[i]
        y = savgol_filter(measure, 51, 3) if smooth_curve_i else measure
        ax.plot(np.arange(len(measure)), y, color=palette[i], label=label)
        if variances:
            ax.fill_between(np.arange(len(measure)), y - (np.array(variances[i])/2), y + (np.array(variances[i])/2), color=palette[i], alpha=0.5)

    x_label = 'Delay Range' if measure_type == 'change_detection' else 'Time Step $t$'
    ax.set_xlabel(x_label, size=font_size, labelpad=1.6)
    ax.set_ylabel(measure_name, size=font_size, labelpad=1.6)
    plt.legend()
    plt.margins(0.01, 0.01)
    plt.tight_layout()
    return ax


def scatter(measures, labels, measure_name, measure_type, layout, fig_size=(10, 5), share_x=True, share_y=True):
    """
    Creates a scatter plot.

    Args:
        measures (list[list]): the list of lists of measures to be visualized
        labels (list[str]): the list of labels for the models
        measure_name (str): the measure to be plotted
        measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        layout (int, int): the layout of the figure (nrows, ncols)
        fig_size (float, float): the figure size of the plot
        share_x (bool): True if the x axis should be shared among plots in the figure, False otherwise
        share_y (bool): True if the y axis among plots in the figure, False otherwise

    Returns:
        Axes: the Axes object(s) containing the scatter plot(s)
    """
    if not measure_type == 'prediction':
        warnings.warn(f'Only measures of type "prediction" can be visualized with method scatter.')
        return

    n_measures = len(measures)
    if layout[0] * layout[1] < n_measures:
        warnings.warn('The number of measures cannot be plotted in such a layout.')
        return

    # noinspection PyTypeChecker
    fig, axes = plt.subplots(layout[0], layout[1], figsize=fig_size, sharex=share_x, sharey=share_y)
    for i in range(layout[0]):
        for j in range(layout[1]):
            ax = axes if n_measures == 1 else (axes[i + j] if layout[0] == 1 or layout[1] == 1 else axes[i, j])
            ax.scatter(np.arange(len(measures[i + j])), measures[i + j], color=palette[i + j],
                       label=labels[i + j])
            ax.set_xlabel('Time Step $t$', size=font_size, labelpad=1.6)
            ax.set_ylabel(measure_name, size=font_size, labelpad=1.6)
            ax.legend()
    plt.margins(0.01, 0.01)
    plt.tight_layout()
    return axes


def bar(measures, labels, measure_name, measure_type, fig_size=(10, 5)):
    """
    Creates a bar plot.

    Args:
        measures (list[list]): the list of lists of measures to be visualized
        labels (list[str]): the list of labels for the models
        measure_name (str): the measure to be plotted
        measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        fig_size (float, float): the figure size of the plot

    Returns:
        Axes: the Axes object containing the bar plot
    """
    if not measure_type == 'prediction':
        warnings.warn(f'Only measures of type "prediction" can be visualized with method bar.')
        return

    fig, ax = plt.subplots(figsize=fig_size)
    width = 0.8
    n_measures = float(len(measures))
    for i, (measure, label) in enumerate(zip(measures, labels)):
        ax.bar(np.arange(len(measure)) - width / 2. + i / n_measures * width, measure, width=width / n_measures,
               align="edge", color=palette[i], label=label)
    ax.set_xlabel('Time Step $t$', size=font_size, labelpad=1.6)
    ax.set_ylabel(measure_name, size=font_size, labelpad=1.6)
    plt.legend()
    plt.margins(0.01, 0.01)
    plt.tight_layout()
    return ax


def selected_features_scatter(measures, labels, measure_type, layout, fig_size=(10, 5), share_x=True, share_y=True):
    """
    Draws the selected features at each time step in a scatter plot.

    Args:
        measures (list[list]): the list of lists of measures to be visualized
        labels (list[str]): the list of labels for the models
        measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        layout (int, int): the layout of the figure (nrows, ncols)
        fig_size (float, float): the figure size of the plot
        share_x (bool): True if the x axis should be shared among plots in the figure, False otherwise
        share_y (bool): True if the y axis among plots in the figure, False otherwise

    Returns:
        Axes: the Axes object containing the scatter plot
    """
    if not measure_type == 'feature_selection':
        warnings.warn(
            f'Only measures of type "feature_selection" can be visualized with method draw_selected_features.')
        return

    n_measures = len(measures)
    if layout[0] * layout[1] < n_measures:
        warnings.warn('The number of measures cannot be plotted in such a layout.')
        return

    # noinspection PyTypeChecker
    fig, axes = plt.subplots(layout[0], layout[1], figsize=fig_size, sharex=share_x, sharey=share_y)
    for i in range(layout[0]):
        for j in range(layout[1]):
            x, y = [], []
            for k, val in enumerate(measures[i + j]):
                x.extend(np.ones(len(val), dtype=int) * k)
                y.extend(val)

            ax = axes if n_measures == 1 else (axes[i + j] if layout[0] == 1 or layout[1] == 1 else axes[i, j])
            ax.grid(True)
            ax.set_xlabel('Time Step $t$', size=font_size, labelpad=1.6)
            ax.set_ylabel('Feature Index', size=font_size, labelpad=1.5)
            ax.tick_params(axis='both', labelsize=font_size * 0.7, length=0)
            ax.scatter(x, y, marker='.', zorder=100, color=palette[i + j], label=labels[i + j])
            ax.legend(frameon=True, loc='best', fontsize=font_size * 0.7, borderpad=0.2, handletextpad=0.2)
    plt.margins(0.01, 0.01)
    plt.tight_layout()
    return axes


def top_features_bar(measures, labels, measure_type, feature_names, layout, fig_size=(10, 5), share_x=True, share_y=True):
    """
    Draws the most selected features over time as a bar plot.

    Args:
        measures (list[list]): the list of lists of measures to be visualized
        labels (list[str]): the list of labels for the models
        measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        feature_names (list): the list of feature names
        layout (int, int): the layout of the figure (nrows, ncols)
        fig_size (float, float): the figure size of the plot
        share_x (bool): True if the x axis should be shared among plots in the figure, False otherwise
        share_y (bool): True if the y axis among plots in the figure, False otherwise

    Returns:
        Axes: the Axes object containing the bar plot
    """
    if not measure_type == 'feature_selection':
        warnings.warn(f'Only measures of type "feature_selection" can be visualized with method draw_top_features.')
        return

    n_measures = len(measures)
    if layout[0] * layout[1] < n_measures:
        warnings.warn('The number of measures cannot be plotted in such a layout.')
        return

    # noinspection PyTypeChecker
    fig, axes = plt.subplots(layout[0], layout[1], figsize=fig_size, sharex=share_x, sharey=share_y)
    for i in range(layout[0]):
        for j in range(layout[1]):
            n_selected_features = len(measures[i + j][0])
            y = [feature for features in measures[i + j] for feature in features]
            counts = [(x, y.count(x)) for x in np.unique(y)]
            top_features = sorted(counts, key=lambda x: x[1])[-n_selected_features:][::-1]
            top_features_idx = [x[0] for x in top_features]
            top_features_vals = [x[1] for x in top_features]

            ax = axes if n_measures == 1 else (axes[i + j] if layout[0] == 1 or layout[1] == 1 else axes[i, j])
            ax.grid(True, axis='y')
            ax.bar(np.arange(n_selected_features), top_features_vals, width=0.3, zorder=100,
                   color=palette[i + j], label=labels[i + j])
            ax.set_xticks(np.arange(n_selected_features))
            ax.set_xticklabels(np.asarray(feature_names)[top_features_idx], rotation=20, ha='right')
            ax.set_ylabel('Times Selected', size=font_size, labelpad=1.5)
            ax.set_xlabel('Top 10 Features', size=font_size, labelpad=1.6)
            ax.tick_params(axis='both', labelsize=font_size * 0.7, length=0)
            ax.legend()
    plt.margins(0.01, 0.01)
    return axes


def top_features_reference_bar(measures, labels, measure_type, feature_names, fig_size=(10, 5)):
    """
    Draws the most selected features over time as a bar plot.

    Args:
        measures (list[list]): the list of lists of measures to be visualized
        labels (list[str]): the list of labels for the models
        measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        feature_names (list): the list of feature names
        fig_size (float, float): the figure size of the plot

    Returns:
        Axes: the Axes object containing the bar plot
    """
    if not measure_type == 'feature_selection':
        warnings.warn(f'Only measures of type "feature_selection" can be visualized with method draw_top_features.')
        return

    width = 0.8
    n_measures = len(measures)

    fig, ax = plt.subplots(figsize=fig_size)
    for i, (measure, label) in enumerate(zip(measures, labels)):
        n_selected_features = len(measure[0])
        y = [feature for features in measure for feature in features]
        counts = [(x, y.count(x)) for x in np.unique(y)]
        if i == 0:
            top_features = sorted(counts, key=lambda x: x[1])[-n_selected_features:][::-1]
            top_features_idx = [x[0] for x in top_features]
            top_features_vals = [x[1] for x in top_features]
        else:
            top_features_vals = [dict(counts)[x] if x in dict(counts).keys() else 0 for x in top_features_idx]
            print(
                f"Top {label} features not in reference {labels[0]}: {np.asarray(feature_names)[[x for x in top_features_idx if x not in dict(counts).keys()]]}")

        ax.grid(True, axis='y')
        ax.bar(np.arange(n_selected_features) - width / 2. + i / n_measures * width, top_features_vals,
               width=width / n_measures, zorder=100, color=palette[i], label=label)
    plt.xticks(np.arange(n_selected_features), labels=np.asarray(feature_names)[top_features_idx], rotation=20, ha='right')
    plt.ylabel('Times Selected', size=font_size, labelpad=1.5)
    plt.xlabel('Top 10 Features', size=font_size, labelpad=1.6)
    plt.tick_params(axis='both', labelsize=font_size * 0.7, length=0)
    plt.legend()
    plt.margins(0.01, 0.01)
    plt.tight_layout()
    return ax


def concept_drifts_scatter(measures, labels, measure_type, data_stream, known_drifts, batch_size, fig_size=(10, 5)):
    """
    Draws the known and the detected concept drifts for all concept drift detectors.

    Args:
        measures (list[list]): the list of lists of measures to be visualized
        labels (list[str]): the list of labels for the models
        measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        data_stream (Stream): the data set as a stream
        known_drifts (list): the known concept drifts for this data set
        batch_size (int): the batch size used for evaluation of the data stream
        fig_size (float, float): the figure size of the plot

    Returns:
        Axes: the Axes object containing the bar plot
    """
    if not measure_type == 'change_detection':
        warnings.warn(f'Only measures of type "change_detection" can be visualized with method draw_concept_drifts.')
        return

    n_measures = len(measures)
    fig, ax = plt.subplots(figsize=fig_size)

    # Draw known drifts
    for known_drift in known_drifts:
        if isinstance(known_drift, tuple):
            ax.axvspan(known_drift[0], known_drift[1], facecolor='#eff3ff', edgecolor='#9ecae1', hatch="//")
        else:
            ax.axvline(known_drift, color=palette[1], lw=3, zorder=0)

    # Draw detected drifts
    y_loc = 0
    y_tick_labels = []
    for measure, label in zip(measures, labels):
        detected_drifts = np.asarray(measure) * batch_size + batch_size
        ax.axhline(y_loc, color=palette[0], zorder=5)
        ax.scatter(detected_drifts, np.repeat(y_loc, len(detected_drifts)), marker='|', color=palette[0], s=300,
                   zorder=10)
        y_loc += (1 / n_measures)
        y_tick_labels.append(label)

    plt.yticks(np.arange(0, 1, 1 / n_measures), y_tick_labels, fontsize=12)
    plt.xticks(np.arange(0, data_stream.n_samples - 10, round(data_stream.n_samples * 0.1)), fontsize=12)
    plt.xlim(-data_stream.n_samples * 0.005, data_stream.n_samples + data_stream.n_samples * 0.005)
    plt.xlabel('# Observations', fontsize=14)
    known_drift_patch = mlines.Line2D([], [], marker='|', linestyle='None', markersize=10, markeredgewidth=2,
                                      color=palette[1], label='known drift')
    detected_drift_patch = mlines.Line2D([], [], marker='|', linestyle='None', markersize=10, markeredgewidth=2,
                                         color=palette[0], label='detected drift')
    plt.legend(handles=[known_drift_patch, detected_drift_patch])
    plt.tight_layout()
    return ax


def spider_chart(measures, labels, measure_names):
    """
    Draws a spider chart of the given measure values and models.

    Args:
        measures (list[list]): the list of lists containing the measure values for each model
        labels (list[str]): the list of labels for the models
        measure_names (list[str]): the measures to be plotted

    Returns:
        Axes: the Axes object containing the bar plot
    """
    N = len(measures[0])
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], measure_names)
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.125, 0.25], ["0", "0.125", "0.25"], color=palette[-1], size=7)
    plt.ylim(0, 0.25)

    for i, (measure, label) in enumerate(zip(measures, labels)):
        measure += measure[:1]
        ax.plot(angles, measure, color=palette[i], linewidth=1, linestyle='solid', label=label)
        ax.fill(angles, measure, color=palette[i], alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.margins(0.1, 0.1)
    plt.tight_layout()
    return ax
