import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import warnings
from scipy.signal import savgol_filter


class Visualizer:
    """
    Class for creating plots to visualize information.

    """
    def __init__(self, measures, labels, measure_type):
        """
        Initialize the visualizer using a uniform style.

        Args:
            measures (list[list]): the list of lists of measures to be visualized
            labels (list[str]): the list of labels for the measures
            measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
        """
        self.measures = measures
        self.labels = labels
        self.measure_type = measure_type
        self.palette = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03',
                        '#ae2012', '#9b2226']
        self.font_size = 12

    def plot(self, plot_title, fig_size=(10.2, 5.2), smooth_curve=False):
        """
        Creates a line plot.

        Args:
            plot_title (str): the title of the plot
            fig_size (float, float):

        Returns:
            Axes: the Axes object containing the line plot
        """
        if not self.measure_type == 'prediction':
            warnings.warn(f'Only measures of type "prediction" can be visualized with method plot.')
            return

        fig, ax = plt.subplots(figsize=fig_size)
        for i, (measure, label) in enumerate(zip(self.measures, self.labels)):
            y = savgol_filter(measure, 51, 3) if smooth_curve else measure
            ax.plot(np.arange(len(measure)), y, color=self.palette[i], label=label)
        ax.set_xlabel('Time Step $t$', size=self.font_size, labelpad=1.6)
        ax.set_ylabel('Metric Value', size=self.font_size, labelpad=1.6)
        plt.legend()
        plt.title(plot_title)
        return ax

    def scatter(self, plot_title, layout, fig_size=(10, 5), share_x=True, share_y=True):
        """
        Creates a scatter plot.

        Args:
            plot_title (str): the title of the plot
            layout (int, int): the layout of the figure (nrows, ncols)
            fig_size (float, float): the figure size of the plot
            share_x (bool): True if the x axis should be shared among plots in the figure, False otherwise
            share_y (bool): True if the y axis among plots in the figure, False otherwise

        Returns:
            Axes: the Axes object(s) containing the scatter plot(s)
        """
        if not self.measure_type == 'prediction':
            warnings.warn(f'Only measures of type "prediction" can be visualized with method scatter.')
            return

        n_measures = len(self.measures)
        if layout[0] * layout[1] < n_measures:
            warnings.warn('The number of measures cannot be plotted in such a layout.')
            return

        fig, axes = plt.subplots(layout[0], layout[1], figsize=fig_size, sharex=share_x, sharey=share_y)
        for i in range(layout[0]):
            for j in range(layout[1]):
                ax = axes if n_measures == 1 else (axes[i + j] if layout[0] == 1 or layout[1] == 1 else axes[i, j])
                ax.scatter(np.arange(len(self.measures[i + j])), self.measures[i + j], color=self.palette[i + j],
                           label=self.labels[i + j])
                ax.set_xlabel('Time Step $t$', size=self.font_size, labelpad=1.6)
                ax.set_ylabel('Metric Value', size=self.font_size, labelpad=1.6)
                ax.legend()
        plt.suptitle(plot_title)
        return axes

    def bar(self, plot_title, fig_size=(10, 5)):
        """
        Creates a bar plot.

        Args:
            plot_title (str): the title of the plot
            fig_size (float, float): the figure size of the plot

        Returns:
            Axes: the Axes object containing the bar plot
        """
        if not self.measure_type == 'prediction':
            warnings.warn(f'Only measures of type "prediction" can be visualized with method bar.')
            return

        fig, ax = plt.subplots(figsize=fig_size)
        width = 0.8
        n_measures = float(len(self.measures))
        for i, (measure, label) in enumerate(zip(self.measures, self.labels)):
            ax.bar(np.arange(len(measure)) - width / 2. + i / n_measures * width, measure, width=width / n_measures,
                   align="edge", color=self.palette[i], label=label)
        ax.set_xlabel('Time Step $t$', size=self.font_size, labelpad=1.6)
        ax.set_ylabel('Metric Value', size=self.font_size, labelpad=1.6)
        plt.legend()
        plt.title(plot_title)
        return ax

    def draw_selected_features(self, layout, fig_size=(10, 5), share_x=True, share_y=True):
        """
        Draws the selected features at each time step in a scatter plot.

        Args:
            layout (int, int): the layout of the figure (nrows, ncols)
            fig_size (float, float): the figure size of the plot
            share_x (bool): True if the x axis should be shared among plots in the figure, False otherwise
            share_y (bool): True if the y axis among plots in the figure, False otherwise

        Returns:
            Axes: the Axes object containing the scatter plot
        """
        if not self.measure_type == 'feature_selection':
            warnings.warn(
                f'Only measures of type "feature_selection" can be visualized with method draw_selected_features.')
            return

        n_measures = len(self.measures)
        if layout[0] * layout[1] < n_measures:
            warnings.warn('The number of measures cannot be plotted in such a layout.')
            return

        fig, axes = plt.subplots(layout[0], layout[1], figsize=fig_size, sharex=share_x, sharey=share_y)
        for i in range(layout[0]):
            for j in range(layout[1]):
                x, y = [], []
                for k, val in enumerate(self.measures[i + j]):
                    x.extend(np.ones(len(val), dtype=int) * k)
                    y.extend(val)

                ax = axes if n_measures == 1 else (axes[i + j] if layout[0] == 1 or layout[1] == 1 else axes[i, j])
                ax.grid(True)
                ax.set_xlabel('Time Step $t$', size=self.font_size, labelpad=1.6)
                ax.set_ylabel('Feature Index', size=self.font_size, labelpad=1.5)
                ax.tick_params(axis='both', labelsize=self.font_size * 0.7, length=0)
                ax.scatter(x, y, marker='.', zorder=100, color=self.palette[i + j], label=self.labels[i + j])
                ax.legend(frameon=True, loc='best', fontsize=self.font_size * 0.7, borderpad=0.2, handletextpad=0.2)
        plt.suptitle('Selected Features At Each Time Step', size=self.font_size)
        return axes

    def draw_top_features(self, feature_names, layout, fig_size=(10, 5), share_x=True, share_y=True):
        """
        Draws the most selected features over time as a bar plot.

        Args:
            feature_names (list): the list of feature names
            layout (int, int): the layout of the figure (nrows, ncols)
            fig_size (float, float): the figure size of the plot
            share_x (bool): True if the x axis should be shared among plots in the figure, False otherwise
            share_y (bool): True if the y axis among plots in the figure, False otherwise

        Returns:
            Axes: the Axes object containing the bar plot
        """
        if not self.measure_type == 'feature_selection':
            warnings.warn(f'Only measures of type "feature_selection" can be visualized with method draw_top_features.')
            return

        n_measures = len(self.measures)
        if layout[0] * layout[1] < n_measures:
            warnings.warn('The number of measures cannot be plotted in such a layout.')
            return

        fig, axes = plt.subplots(layout[0], layout[1], figsize=fig_size, sharex=share_x, sharey=share_y)
        for i in range(layout[0]):
            for j in range(layout[1]):
                n_selected_features = len(self.measures[i + j][0])
                y = [feature for features in self.measures[i + j] for feature in features]
                counts = [(x, y.count(x)) for x in np.unique(y)]
                top_features = sorted(counts, key=lambda x: x[1])[-n_selected_features:][::-1]
                top_features_idx = [x[0] for x in top_features]
                top_features_vals = [x[1] for x in top_features]

                ax = axes if n_measures == 1 else (axes[i + j] if layout[0] == 1 or layout[1] == 1 else axes[i, j])
                ax.grid(True, axis='y')
                ax.bar(np.arange(n_selected_features), top_features_vals, width=0.3, zorder=100,
                       color=self.palette[i + j], label=self.labels[i + j])
                ax.set_xticks(np.arange(n_selected_features))
                ax.set_xticklabels(np.asarray(feature_names)[top_features_idx], rotation=20, ha='right')
                ax.set_ylabel('Times Selected', size=self.font_size, labelpad=1.5)
                ax.set_xlabel('Top 10 Features', size=self.font_size, labelpad=1.6)
                ax.tick_params(axis='both', labelsize=self.font_size * 0.7, length=0)
                ax.set_xlim(-0.2, 9.2)
                ax.legend()
        plt.suptitle(f'Most Selected Features', size=self.font_size)
        return axes

    def draw_top_features_with_reference(self, feature_names, fig_size=(10, 5)):
        """
        Draws the most selected features over time as a bar plot.

        Args:
            feature_names (list): the list of feature names
            fig_size (float, float): the figure size of the plot

        Returns:
            Axes: the Axes object containing the bar plot
        """
        if not self.measure_type == 'feature_selection':
            warnings.warn(f'Only measures of type "feature_selection" can be visualized with method draw_top_features.')
            return

        width = 0.8
        n_measures = len(self.measures)

        fig, ax = plt.subplots(figsize=fig_size)
        for i, (measure, label) in enumerate(zip(self.measures, self.labels)):
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
                    f"Top {label} features not in reference {self.labels[0]}: {np.asarray(feature_names)[[x for x in top_features_idx if x not in dict(counts).keys()]]}")

            ax.grid(True, axis='y')
            ax.bar(np.arange(n_selected_features) - width / 2. + i / n_measures * width, top_features_vals,
                   width=width / n_measures, zorder=100, color=self.palette[i], label=label)
        plt.xticks(np.arange(n_selected_features), labels=np.asarray(feature_names)[top_features_idx], rotation=20, ha='right')
        plt.ylabel('Times Selected', size=self.font_size, labelpad=1.5)
        plt.xlabel('Top 10 Features', size=self.font_size, labelpad=1.6)
        plt.tick_params(axis='both', labelsize=self.font_size * 0.7, length=0)
        plt.legend()
        plt.title(f'Most Selected Features', size=self.font_size)
        return ax

    def draw_concept_drifts(self, data, known_drifts, batch_sizes, data_labels, cdd_labels):
        if not self.measure_type == 'drift_detection':
            warnings.warn(f'Only measures of type "drift_detection" can be visualized with method draw_concept_drifts.')
            return

        drifts = []
        for i, (stream, known_drift, batch_size, data_label) in enumerate(
                zip(data, known_drifts, batch_sizes, data_labels)):
            drifts.append([])
            for measure, cdd_label in zip(self.measures, cdd_labels):
                drifts[i].append((cdd_label, np.asarray(measure) * batch_size + batch_size))

            fig = plt.figure(figsize=(10, 5))
            y_loc = 0  # y-coordinate of scatter points

            # Draw known drifts
            for true in known_drift:
                if isinstance(true, tuple):
                    plt.axvspan(true[0], true[1], facecolor='#eff3ff', edgecolor='#9ecae1', hatch="//")
                else:
                    plt.axvline(true, color='#9ecae1', lw=3, zorder=0)

            # Draw detected drifts
            y_tick_labels = []

            for model in drifts[i]:
                plt.axhline(y_loc, color='black', zorder=5)
                plt.scatter(model[1], np.repeat(y_loc, len(model[1])), marker='|', color='black', s=300, zorder=10)
                y_loc += 0.5
                y_tick_labels.append(model[0].upper())  # save model name

            plt.yticks(np.arange(0, 1, 0.5), y_tick_labels, fontsize=12)
            tx = np.arange(0, stream.n_samples - 10, round(stream.n_samples * 0.1))
            plt.xticks(tx, tx, fontsize=12)
            plt.xlim(-stream.n_samples * 0.005,
                     stream.n_samples + stream.n_samples * 0.005)
            plt.xlabel('# Observations', fontsize=14)
            plt.title(data_label)
            plt.show()
