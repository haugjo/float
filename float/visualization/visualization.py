import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import warnings
from scipy.signal import savgol_filter
from skmultiflow.data.data_stream import Stream

palette = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03',
           '#ae2012', '#9b2226']
font_size = 12


def plot(measures, labels, measure_name, measure_type, plot_title, fig_size=(10.2, 5.2), smooth_curve=False):
    """
    Creates a line plot.

    Args:
        measures (list[list]): the list of lists of measures to be visualized
        labels (list[str]): the list of labels for the measures
        measure_name (str): the measure to be plotted
        measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        plot_title (str): the title of the plot
        fig_size (float, float): the figure size of the plot
        smooth_curve (bool | list[bool]): True if the plotted curve should be smoothed, False otherwise

    Returns:
        Axes: the Axes object containing the line plot
    """
    if measure_type not in ['prediction', 'change_detection']:
        warnings.warn(f'Only measures of type "prediction" or "change_detection" can be visualized with method plot.')
        return

    fig, ax = plt.subplots(figsize=fig_size)
    for i, (measure, label) in enumerate(zip(measures, labels)):
        smooth_curve_i = smooth_curve if type(smooth_curve) is bool else smooth_curve[i]
        y = savgol_filter(measure, 51, 3) if smooth_curve_i else measure
        ax.plot(np.arange(len(measure)), y, color=palette[i], label=label)

    x_label = 'Delay Range' if measure_type == 'change_detection' else 'Time Step $t$'
    ax.set_xlabel(x_label, size=font_size, labelpad=1.6)
    ax.set_ylabel(measure_name, size=font_size, labelpad=1.6)
    plt.legend()
    plt.title(plot_title)
    return ax


def scatter(measures, labels, measure_name, measure_type, layout, plot_title, fig_size=(10, 5), share_x=True, share_y=True):
    """
    Creates a scatter plot.

    Args:
        measures (list[list]): the list of lists of measures to be visualized
        labels (list[str]): the list of labels for the measures
        measure_name (str): the measure to be plotted
        measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        layout (int, int): the layout of the figure (nrows, ncols)
        plot_title (str): the title of the plot
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
    plt.suptitle(plot_title)
    return axes


def bar(measures, labels, measure_name, measure_type, plot_title, fig_size=(10, 5)):
    """
    Creates a bar plot.

    Args:
        measures (list[list]): the list of lists of measures to be visualized
        labels (list[str]): the list of labels for the measures
        measure_name (str): the measure to be plotted
        measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        plot_title (str): the title of the plot
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
    plt.title(plot_title)
    return ax


def draw_selected_features(measures, labels, measure_type, layout, plot_title, fig_size=(10, 5), share_x=True, share_y=True):
    """
    Draws the selected features at each time step in a scatter plot.

    Args:
        measures (list[list]): the list of lists of measures to be visualized
        labels (list[str]): the list of labels for the measures
        measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        layout (int, int): the layout of the figure (nrows, ncols)
        plot_title (str): the title of the plot
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
    plt.suptitle(plot_title, size=font_size)
    return axes


def draw_top_features(measures, labels, measure_type, feature_names, layout, plot_title, fig_size=(10, 5), share_x=True, share_y=True):
    """
    Draws the most selected features over time as a bar plot.

    Args:
        measures (list[list]): the list of lists of measures to be visualized
        labels (list[str]): the list of labels for the measures
        measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        feature_names (list): the list of feature names
        layout (int, int): the layout of the figure (nrows, ncols)
        plot_title (str): the title of the plot
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
            ax.set_xlim(-0.2, 9.2)
            ax.legend()
    plt.suptitle(plot_title, size=font_size)
    return axes


def draw_top_features_with_reference(measures, labels, measure_type, feature_names, plot_title, fig_size=(10, 5)):
    """
    Draws the most selected features over time as a bar plot.

    Args:
        measures (list[list]): the list of lists of measures to be visualized
        labels (list[str]): the list of labels for the measures
        measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        feature_names (list): the list of feature names
        plot_title (str): the title of the plot
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
    plt.title(plot_title, size=font_size)
    return ax


def draw_concept_drifts(measures, labels, measure_type, data_stream, known_drifts, batch_size, plot_title, fig_size=(10, 5)):
    """
    Draws the known and the detected concept drifts for all concept drift detectors.

    Args:
        measures (list[list]): the list of lists of measures to be visualized
        labels (list[str]): the list of labels for the measures
        measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        data_stream (Stream): the data set as a stream
        known_drifts (list): the known concept drifts for this data set
        batch_size (int): the batch size used for evaluation of the data stream
        plot_title (str): the title of the plot
        fig_size (float, float): the figure size of the plot

    Returns:
        Axes: the Axes object containing the bar plot
    """
    if not measure_type == 'drift_detection':
        warnings.warn(f'Only measures of type "drift_detection" can be visualized with method draw_concept_drifts.')
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
    plt.title(plot_title)
    return ax
