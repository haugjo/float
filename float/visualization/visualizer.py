import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import warnings


class Visualizer:
    """
    Class for creating plots to visualize information.

    """
    def __init__(self, measures, labels, measure_type, fig_size=(10, 5)):
        """
        Initialize the visualizer using a uniform style.

        Args:
            measures (list[list]): the list of lists of measures to be visualized
            labels (list[str]): the list of labels for the measures
            measure_type (str): the type of the measures passed, one of 'prediction', 'feature_selection, or 'drift_detection'
            fig_size (float, float): the size of the plots
        """
        self.measures = measures
        self.labels = labels
        self.measure_type = measure_type
        self.fig_size = fig_size
        self.palette = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']
        self.font_size = 12

    def plot(self, plot_title):
        """
        Creates a line plot.

        Args:
            plot_title (str): the title of the plot

        Returns:
            Axes: the Axes object containing the line plot
        """
        if not self.measure_type == 'prediction':
            warnings.warn(f'Only measures of type "prediction" can be visualized with method plot.')
            return
        fig, ax = plt.subplots(figsize=self.fig_size)
        for i, (measure, label) in enumerate(zip(self.measures, self.labels)):
            print('here')
            ax.plot(np.arange(len(measure)), measure, color=self.palette[i], label=label)
        ax.set_xlabel('Time Step $t$', size=self.font_size, labelpad=1.6)
        ax.set_ylabel('Metric Value', size=self.font_size, labelpad=1.6)
        plt.legend()
        plt.title(plot_title)
        return ax

    def scatter(self, plot_title):
        """
        Creates a scatter plot.

        Args:
            plot_title (str): the title of the plot

        Returns:
            Axes: the Axes object(s) containing the scatter plot(s)
        """
        if not self.measure_type == 'prediction':
            warnings.warn(f'Only measures of type "prediction" can be visualized with method scatter.')
            return
        fig, axes = plt.subplots(len(self.measures), 1, sharex=True, figsize=self.fig_size)
        for i, (measure, label) in enumerate(zip(self.measures, self.labels)):
            ax = axes[i] if len(self.measures) > 1 else axes
            ax.scatter(np.arange(len(measure)), measure, color=self.palette[i], label=label)
            ax.set_ylabel('Metric Value', size=self.font_size, labelpad=1.6)
            ax.legend()
        plt.xlabel('Time Step $t$', size=self.font_size, labelpad=1.6)
        plt.suptitle(plot_title)
        return axes

    def bar(self, plot_title):
        """
        Creates a bar plot.

        Args:
            plot_title (str): the title of the plot

        Returns:
            Axes: the Axes object containing the bar plot
        """
        if not self.measure_type == 'prediction':
            warnings.warn(f'Only measures of type "prediction" can be visualized with method bar.')
            return
        fig, ax = plt.subplots(figsize=self.fig_size)
        width = 0.8
        n_measures = float(len(self.measures))
        for i, (measure, label) in enumerate(zip(self.measures, self.labels)):
            ax.bar(np.arange(len(measure)) - width / 2. + i / n_measures * width, measure, width=width / n_measures, align="edge", color=self.palette[i], label=label)
        ax.set_xlabel('Time Step $t$', size=self.font_size, labelpad=1.6)
        ax.set_ylabel('Metric Value', size=self.font_size, labelpad=1.6)
        plt.legend()
        plt.title(plot_title)
        return ax

    def draw_selected_features(self):
        """
        Draws the selected features at each time step in a scatter plot.

        Returns:
            Axes: the Axes object containing the scatter plot
        """
        if not self.measure_type == 'feature_selection':
            warnings.warn(f'Only measures of type "feature_selection" can be visualized with method draw_selected_features.')
            return
        x, y = [], []
        for i, val in enumerate(self.measures):
            x.extend(np.ones(len(val), dtype=int) * i)
            y.extend(val)

        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.grid(True)
        ax.set_ylabel('Feature Index', size=self.font_size, labelpad=1.5)
        ax.set_xlabel('Time Step $t$', size=self.font_size, labelpad=1.6)
        ax.tick_params(axis='both', labelsize=self.font_size * 0.7, length=0)
        ax.scatter(x, y, marker='.', label='Selected Features', zorder=100, color=self.palette[0])
        plt.legend(frameon=True, loc='best', fontsize=self.font_size * 0.7, borderpad=0.2, handletextpad=0.2)
        plt.title('Selected Features At Each Time Step', size=self.font_size)
        plt.plot()
        return ax

    def draw_top_features(self, feature_names):
        """
        Draws the most selected features over time as a bar plot.

        Args:
            feature_names (list): the list of feature names

        Returns:
            Axes: the Axes object containing the bar plot
        """
        if not self.measure_type == 'feature_selection':
            warnings.warn(f'Only measures of type "feature_selection" can be visualized with method draw_top_features.')
            return
        n_selected_features = len(self.measures[0])
        y = [feature for features in self.measures for feature in features]
        counts = np.bincount(y)
        top_ftr_idx = counts.argsort()[-n_selected_features:][::-1]

        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.grid(True, axis='y')
        ax.bar(np.arange(n_selected_features), counts[top_ftr_idx], width=0.3, zorder=100, color=self.palette[0])
        ax.set_xticks(np.arange(n_selected_features))
        ax.set_xticklabels(np.asarray(feature_names)[top_ftr_idx], rotation=20, ha='right')
        ax.set_ylabel('Times Selected', size=self.font_size, labelpad=1.5)
        ax.set_xlabel('Top 10 Features', size=self.font_size, labelpad=1.6)
        ax.tick_params(axis='both', labelsize=self.font_size * 0.7, length=0)
        ax.set_xlim(-0.2, 9.2)
        plt.title(f'Top {n_selected_features} Most Selected Features', size=self.font_size)
        return ax
