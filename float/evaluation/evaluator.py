from abc import ABCMeta, abstractmethod


class Evaluator(metaclass=ABCMeta):
    """
    Abstract base class for evaluation measures and metrics.

    Attributes:
        measures (list): list of measure values corresponding to the time steps
        line_plot (bool): True if the metric can be visualized as a line plot, False otherwise
        scatter_plot (bool): True if the metric can be visualized as a scatter plot, False otherwise
        bar_plot (bool): True if the metric can be visualized as a bar plot, False otherwise
    """
    def __init__(self, line_plot=False, scatter_plot=False, bar_plot=False):
        """

        Args:
            line_plot (bool): True if the metric can be visualized as a line plot, False otherwise
            scatter_plot (bool): True if the metric can be visualized as a scatter plot, False otherwise
            bar_plot (bool): True if the metric can be visualized as a bar plot, False otherwise
        """
        self.measures = []
        self.line_plot = line_plot
        self.scatter_plot = scatter_plot
        self.bar_plot = bar_plot

    @abstractmethod
    def compute(self, **kwargs):
        """
        Compute measure given inputs at current time step and append self.measures.
        """
        raise NotImplementedError
