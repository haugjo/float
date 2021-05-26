from float.evaluation.evaluator import Evaluator
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import warnings


class Visualizer:
    """
    Class for creating plots to visualize information.
    """
    def __init__(self, evaluator, fig_size):
        """
        Initialize the visualizer using a uniform style.

        Args:
            evaluator (Evaluator): the evaluator object
            fig_size (float, float): the size of the plots
        """
        self.evaluator = evaluator
        self.fig_size = fig_size

    def plot(self):
        """
        Creates a line plot.

        Returns:
            Axes: the Axes object containing the line plot
        """
        if self.evaluator.line_plot:
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.plot(range(len(self.evaluator.measures)), self.evaluator.measures)
            return ax
        else:
            warnings.warn("This metric cannot be visualized with a line plot.")

    def scatter(self):
        """
        Creates a scatter plot.

        Returns:
            Axes: the Axes object containing the scatter plot
        """
        if self.evaluator.scatter_plot:
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.scatter(range(len(self.evaluator.measures)), self.evaluator.measures)
            return ax
        else:
            warnings.warn("This metric cannot be visualized with a scatter plot.")

    def bar(self):
        """
        Creates a bar plot.

        Returns:
            Axes: the Axes object containing the bar plot
        """
        if self.evaluator.bar_plot:
            fig, ax = plt.subplots(figsize=self.fig_size)
            ax.bar(range(len(self.evaluator.measures)), self.evaluator.measures)
            return ax
        else:
            warnings.warn("This metric cannot be visualized with a bar plot.")
