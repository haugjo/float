from float.evaluation.evaluator import Evaluator


class Visualizer:
    """
    Class for creating plots to visualize information.

    # TODO: add plots
    """
    def __init__(self, evaluator, fig_size):
        """
        Initialize the visualizer using a uniform style.

        Args:
            evaluator (Evaluator): the evaluator object
            fig_size (float, float): the size of the plots
        """
        self.evaluator = evaluator
        # TODO: create uniform style guide for all plots
        self.fig_size = fig_size
        self.palette = ['#000000', '#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4', '#91cf60', '#1a9850']
        self.font_size = 12
