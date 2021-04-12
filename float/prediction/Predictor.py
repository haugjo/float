from abc import ABCMeta, abstractmethod


class Predictor(metaclass=ABCMeta):
    """
    Abstract base class which serves as both a wrapper for skmultiflow predictive models and a constructor for
    user-defined predictive models.
    """
    @abstractmethod
    def __init__(self):
        """
        Initializes the predictor.
        """
        raise NotImplementedError

    def fit(self, X, y, sample_weight=None):
        """
        Fits the model.

        Args:
            X (np.array): data samples to train the model with
            y (np.array): target values for all samples in X
            sample_weight (np.array): sample weights

        Returns:
            self
        """
        self.partial_fit(X, y, sample_weight=sample_weight)

        return self

    @abstractmethod
    def partial_fit(self, X, y, sample_weight=None):
        """
        Partially (incrementally) fit the model.

        Args:
            X (np.array): data samples to train the model with
            y (np.array): target values for all samples in X
            sample_weight (np.array): sample weights

        Returns:
            self
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """
        Predicts target values for the passed data.

        Args:
            X (np.array): data samples to predict the values for

        Returns:
            np.array: predictions for all samples in X
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X):
        """
        Estimates the probability for probabilistic/bayesian regressors.

        Args:
            X (np.array): data samples to predict the probabilities for

        Returns:
            np.array: prediction probability for all samples in X
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, X, y):
        """
        Returns the accuracy based on the given test samples and true values.

        Args:
            X (np.array): test data samples
            y (np.array): true values for all samples in X

        Returns:
            float: accuracy based on test data and target values
        """
        raise NotImplementedError
