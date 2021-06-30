import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


class Visualizer:
    """
    Class for creating plots to visualize information.

    TODO: figure out how to check if measures can be visualized with that specific plot
    """
    def __init__(self, measures, fig_size=(10, 5)):
        """
        Initialize the visualizer using a uniform style.

        Args:
            measures (list): the list of measures to be visualized
            fig_size (float, float): the size of the plots
        """
        self.measures = measures
        self.fig_size = fig_size
        self.font_size = 12

    def plot(self, metric_name):
        """
        Creates a line plot.

        Args:
            metric_name (str): the name of the metric to plot

        Returns:
            Axes: the Axes object containing the line plot
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.plot(range(len(self.measures)), self.measures)
        ax.set_xlabel('Time Step $t$', size=self.font_size, labelpad=1.6)
        ax.set_ylabel('Metric Value', size=self.font_size, labelpad=1.6)
        plt.title(metric_name)
        return ax

    def scatter(self, metric_name):
        """
        Creates a scatter plot.

        Args:
            metric_name (str): the name of the metric to plot

        Returns:
            Axes: the Axes object containing the scatter plot
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.scatter(range(len(self.measures)), self.measures)
        ax.set_xlabel('Time Step $t$', size=self.font_size, labelpad=1.6)
        ax.set_ylabel('Metric Value', size=self.font_size, labelpad=1.6)
        plt.title(metric_name)
        return ax

    def bar(self, metric_name):
        """
        Creates a bar plot.

        Args:
            metric_name (str): the name of the metric to plot

        Returns:
            Axes: the Axes object containing the bar plot
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.bar(range(len(self.measures)), self.measures)
        ax.set_xlabel('Time Step $t$', size=self.font_size, labelpad=1.6)
        ax.set_ylabel('Metric Value', size=self.font_size, labelpad=1.6)
        plt.title(metric_name)
        return ax

    def draw_selected_features(self):
        """
        Draws the selected features at each time step in a scatter plot.

        Returns:
            Axes: the Axes object containing the scatter plot
        """
        x, y = [], []
        for i, val in enumerate(self.measures):
            x.extend(np.ones(len(val), dtype=int) * i)
            y.extend(val)

        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.grid(True)
        ax.set_ylabel('Feature Index', size=self.font_size, labelpad=1.5)
        ax.set_xlabel('Time Step $t$', size=self.font_size, labelpad=1.6)
        ax.tick_params(axis='both', labelsize=self.font_size * 0.7, length=0)
        ax.scatter(x, y, marker='.', label='Selected Features', zorder=100)  # , color=self.palette[0])
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
        n_selected_features = len(self.measures[0])
        y = [feature for features in self.measures for feature in features]
        counts = np.bincount(y)
        top_ftr_idx = counts.argsort()[-n_selected_features:][::-1]

        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.grid(True, axis='y')
        ax.bar(np.arange(n_selected_features), counts[top_ftr_idx], width=0.3, zorder=100)
        ax.set_xticklabels(np.asarray(feature_names)[top_ftr_idx], rotation=20, ha='right')
        ax.set_ylabel('Times Selected', size=self.font_size, labelpad=1.5)
        ax.set_xlabel('Top 10 Features', size=self.font_size, labelpad=1.6)
        ax.tick_params(axis='both', labelsize=self.font_size * 0.7, length=0)
        ax.set_xticks(np.arange(n_selected_features))
        ax.set_xlim(-0.2, 9.2)
        plt.title(f'Top {n_selected_features} Most Selected Features', size=self.font_size)
        return ax
