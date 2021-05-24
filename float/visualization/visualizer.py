from float.evaluation.evaluator import Evaluator
import matplotlib.pyplot as plt
import numpy as np
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
        Create a line plot.
        """
        if self.evaluator.line_plot:
            plt.figure(figsize=self.fig_size)
            plt.plot(range(len(self.evaluator.measures)), self.evaluator.measures)
            plt.show()
        else:
            warnings.warn("This metric cannot be visualized with a line plot.")

    def scatter(self):
        """
        Create a scatter plot.
        """
        if self.evaluator.scatter_plot:
            plt.figure(figsize=self.fig_size)
            plt.scatter(range(len(self.evaluator.measures)), self.evaluator.measures)
            plt.show()
        else:
            warnings.warn("This metric cannot be visualized with a scatter plot.")

    def bar(self):
        """
        Create a bar plot.
        """
        if self.evaluator.bar_plot:
            plt.figure(figsize=self.fig_size)
            plt.bar(range(len(self.evaluator.measures)), self.evaluator.measures)
            plt.show()
        else:
            warnings.warn("This metric cannot be visualized with a bar plot.")
