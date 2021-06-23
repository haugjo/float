import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


# TODO figure out how to check if measures can be visualized with that specific plot
class Visualizer:
    """
    Class for creating plots to visualize information.
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

    def plot(self):
        """
        Creates a line plot.

        Returns:
            Axes: the Axes object containing the line plot
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.plot(range(len(self.measures)), self.measures)
        return ax

    def scatter(self):
        """
        Creates a scatter plot.

        Returns:
            Axes: the Axes object containing the scatter plot
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.scatter(range(len(self.measures)), self.measures)
        return ax

    def bar(self):
        """
        Creates a bar plot.

        Returns:
            Axes: the Axes object containing the bar plot
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.bar(range(len(self.measures)), self.measures)
        return ax

    def _draw_top_features_plot(self, feature_names):
        """
        Draws the most selected features over time.

        Args:
            feature_names (list): the list of feature names

        Returns:
            Axes: the Axes object containing the bar plot
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        ax.grid(True, axis='y')
        y = [feature for features in self.measures for feature in features]
        counts = np.bincount(y)
        top_ftr_idx = counts.argsort()[-10:][::-1]
        ax.bar(np.arange(10), counts[top_ftr_idx], width=0.3, zorder=100)
        ax.set_xticklabels(np.asarray(feature_names)[top_ftr_idx])
        ax.set_ylabel('Times Selected', size=self.font_size, labelpad=1.5)
        ax.set_xlabel('Top 10 Features', size=self.font_size, labelpad=1.6)
        ax.tick_params(axis='both', labelsize=self.font_size * 0.7, length=0)
        ax.set_xticks(np.arange(10))
        ax.set_xlim(-0.2, 9.2)
        return ax
