from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, zero_one_loss


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
    def __init__(self, classes):
        """
        Initializes the predictor.

        Args:
            classes (list): the list of classes in the data
        """
        self.predictions = []
        self.accuracy_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
        self.losses = []
        self.testing_times = []
        self.training_times = []
        self.classes = classes

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
        y_pred = self.predict(X)
        self.accuracy_scores.append(self._get_accuracy(y, y_pred))
        self.precision_scores.append(self._get_precision(y, y_pred))
        self.recall_scores.append(self._get_recall(y, y_pred))
        self.f1_scores.append(self._get_f1_score(y, y_pred))
        self.losses.append(self._get_loss(y, y_pred))

    @staticmethod
    def _get_accuracy(y_true, y_pred):
        """
        Returns the accuracy based on the given test samples and true values.

        Args:
            y_true (np.ndarray): true values for all samples in X
            y_pred (np.ndarray): predicted values for all samples in X

        Returns:
            float: accuracy based on test data and target values
        """
        return accuracy_score(y_true, y_pred)

    def _get_precision(self, y_true, y_pred):
        """
        Returns the precision based on the given test samples and true values.

        Args:
            y_true (np.ndarray): true values for all samples in X
            y_pred (np.ndarray): predicted values for all samples in X

        Returns:
            float: precision based on test data and target values
        """
        return precision_score(y_true, y_pred, labels=self.classes, average='weighted', zero_division=0)

    def _get_recall(self, y_true, y_pred):
        """
        Returns the recall based on the given test samples and true values.

        Args:
            y_true (np.ndarray): true values for all samples in X
            y_pred (np.ndarray): predicted values for all samples in X

        Returns:
            float: recall based on test data and target values
        """
        return recall_score(y_true, y_pred, labels=self.classes, average='weighted', zero_division=0)

    def _get_f1_score(self, y_true, y_pred):
        """
        Returns the f1 score based on the given test samples and true values.

        Args:
            y_true (np.ndarray): true values for all samples in X
            y_pred (np.ndarray): predicted values for all samples in X

        Returns:
            float: f1 score based on test data and target values
        """
        return f1_score(y_true, y_pred, labels=self.classes, average='weighted', zero_division=0)

    @staticmethod
    def _get_loss(y_true, y_pred):
        """
        Returns the 0-1 loss based on the given test samples and true values.

        Args:
            y_true (np.ndarray): true values for all samples in X
            y_pred (np.ndarray): predicted values for all samples in X

        Returns:
            float: loss based on test data and target values
        """
        return zero_one_loss(y_true, y_pred)
