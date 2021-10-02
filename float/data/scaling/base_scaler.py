from abc import ABCMeta, abstractmethod


class BaseScaler(metaclass=ABCMeta):
    """ Abstract Base Class for Online Scaling and Normalization

        Attributes:
            reset_after_drift (bool): indicates whether to reset the predictor after a drift was detected
    """
    def __init__(self, reset_after_drift):
        """ Initialize the Scaler object

        Args:
            reset_after_drift (bool): indicates whether to reset the predictor after a drift was detected
        """
        self.reset_after_drift = reset_after_drift

    @abstractmethod
    def partial_fit(self, X):
        """ Update/Fit the scaler

        Args:
            X (np.array): data sample
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        """ Scale the given data sample

        Args:
            X (np.array): data sample

        Returns:
            np.array: scaled data sample
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """ Reset the scaler
        """
        raise NotImplementedError