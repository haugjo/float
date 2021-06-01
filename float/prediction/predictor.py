from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Predictor(metaclass=ABCMeta):
    """
    Abstract base class which serves as both a wrapper for skmultiflow predictive models and a constructor for
    user-defined predictive models.

    Attributes:
        predictions (list): predicted labels per time step
        accuracy_scores (list): accuracy scores per time step
        precision_scores (list): precision scores per time step
        recall_scores (list): recall scores per time step
        f1_scores (list): f1 scores per time step
        testing_times (list): testing times per time step
        training_times (list): training times per time step
    """
    @abstractmethod
    def __init__(self):
        """
        Initializes the predictor.
        """
        self.predictions = []
        self.accuracy_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
        self.testing_times = []
        self.training_times = []

    def fit(self, X, y, sample_weight=None):
        """
        Fits the model.

        Args:
            X (np.ndarray): data samples to train the model with
            y (np.ndarray): target values for all samples in X
            sample_weight (np.ndarray): sample weights

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
            X (np.ndarray): data samples to train the model with
            y (np.ndarray): target values for all samples in X
            sample_weight (np.ndarray): sample weights

        Returns:
            self
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """
        Predicts target values for the passed data.

        Args:
            X (np.ndarray): data samples to predict the values for

        Returns:
            np.ndarray: predictions for all samples in X
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X):
        """
        Estimates the probability for probabilistic/bayesian regressors.

        Args:
            X (np.ndarray): data samples to predict the probabilities for

        Returns:
            np.ndarray: prediction probability for all samples in X
        """
        raise NotImplementedError

    def evaluate(self, X, y):
        """
        Evaluates the predictor at one time step.

        Args:
            X (np.ndarray): test data samples
            y (np.ndarray): true values for all samples in X
        """
        self.accuracy_scores.append(self._get_accuracy(X, y))
        self.precision_scores.append(self._get_precision(X, y))
        self.recall_scores.append(self._get_recall(X, y))
        self.f1_scores.append(self._get_f1_score(X, y))

    def _get_accuracy(self, X, y):
        """
        Returns the accuracy based on the given test samples and true values.

        Args:
            X (np.ndarray): test data samples
            y (np.ndarray): true values for all samples in X

        Returns:
            float: accuracy based on test data and target values
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def _get_precision(self, X, y):
        """
        Returns the precision based on the given test samples and true values.

        Args:
            X (np.ndarray): test data samples
            y (np.ndarray): true values for all samples in X

        Returns:
            float: precision based on test data and target values
        """
        y_pred = self.predict(X)
        return precision_score(y, y_pred)

    def _get_recall(self, X, y):
        """
        Returns the recall based on the given test samples and true values.

        Args:
            X (np.ndarray): test data samples
            y (np.ndarray): true values for all samples in X

        Returns:
            float: recall based on test data and target values
        """
        y_pred = self.predict(X)
        return recall_score(y, y_pred)

    def _get_f1_score(self, X, y):
        """
        Returns the f1 score based on the given test samples and true values.

        Args:
            X (np.ndarray): test data samples
            y (np.ndarray): true values for all samples in X

        Returns:
            float: f1 score based on test data and target values
        """
        y_pred = self.predict(X)
        return f1_score(y, y_pred)
