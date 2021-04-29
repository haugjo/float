from abc import ABCMeta, abstractmethod


class Evaluator(metaclass=ABCMeta):
    """
    Abstract base class for evaluation measures and metrics.

    Attributes:
        measures (list): list of measure values corresponding to the time steps
    """
    def __init__(self):
        self.measures = []

    @abstractmethod
    def compute(self, **kwargs):
        """
        Compute measure given inputs at current time step and append self.measures.
        """
        raise NotImplementedError
