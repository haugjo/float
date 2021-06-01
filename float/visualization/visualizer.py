import matplotlib.pyplot as plt
from matplotlib.axes import Axes


# TODO figure out how to check if measures can be visualized with that specific plot
class Visualizer:
    """
    Class for creating plots to visualize information.
    """
    def __init__(self, measures, fig_size):
        """
        Initialize the visualizer using a uniform style.

        Args:
            measures (list): the list of measures to be visualized
            fig_size (float, float): the size of the plots
        """
        self.measures = measures
        self.fig_size = fig_size

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
