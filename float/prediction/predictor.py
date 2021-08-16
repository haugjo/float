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
        accuracy_scores_decay (list): accuracy scores decayed per time step
        precision_scores_decay (list): precision scores decayed per time step
        recall_scores_decay (list): recall scores decayed per time step
        f1_scores_decay (list): f1 scores decayed per time step
        accuracy_scores_window (list): accuracy scores decayed for a sliding window
        precision_scores_window (list): precision scores decayed for a sliding window
        recall_scores_window (list): recall scores decayed for a sliding window
        f1_scores_window (list): f1 scores decayed for a sliding window
        losses (list): 0-1-losses per time step
        testing_times (list): testing times per time step
        training_times (list): training times per time step
    """

    @abstractmethod
    def __init__(self, classes, decay_rate, window_size):
        """
        Initializes the predictor.

        Args:
            classes (list): the list of classes in the data
        """
        self.classes = classes
        self.decay_rate = decay_rate
        self.window_size = window_size

        self.predictions = []
        self.accuracy_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []

        if self.decay_rate:
            self.accuracy_scores_decay = []
            self.precision_scores_decay = []
            self.recall_scores_decay = []
            self.f1_scores_decay = []
        if self.window_size:
            self.accuracy_scores_window = []
            self.precision_scores_window = []
            self.recall_scores_window = []
            self.f1_scores_window = []

        self.losses = []
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
        y_pred = self.predict(X)
        accuracy = self._get_accuracy(y, y_pred)
        self.accuracy_scores.append(accuracy)
        precision = self._get_precision(y, y_pred)
        self.precision_scores.append(precision)
        recall = self._get_recall(y, y_pred)
        self.recall_scores.append(recall)
        f1 = self._get_f1_score(y, y_pred)
        self.f1_scores.append(f1)

        if self.decay_rate:
            self.evaluate_decay(accuracy, f1, precision, recall)

        if self.window_size:
            self.evaluate_window()

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

    def evaluate_decay(self, accuracy, f1, precision, recall):
        if len(self.accuracy_scores_decay) == 0:
            self.accuracy_scores_decay.append(accuracy)
            self.precision_scores_decay.append(precision)
            self.recall_scores_decay.append(recall)
            self.f1_scores_decay.append(f1)
        else:
            self.accuracy_scores_decay.append(
                self.decay_rate * accuracy + (1 - self.decay_rate) * self.accuracy_scores_decay[-1])
            self.precision_scores_decay.append(
                self.decay_rate * precision + (1 - self.decay_rate) * self.precision_scores_decay[-1])
            self.recall_scores_decay.append(
                self.decay_rate * recall + (1 - self.decay_rate) * self.recall_scores_decay[-1])
            self.f1_scores_decay.append(self.decay_rate * f1 + (1 - self.decay_rate) * self.f1_scores_decay[-1])

    def evaluate_window(self):
        self.accuracy_scores_window.append(np.mean(self.accuracy_scores[-self.window_size:]))
        self.precision_scores_window.append(np.mean(self.precision_scores[-self.window_size:]))
        self.recall_scores_window.append(np.mean(self.recall_scores[-self.window_size:]))
        self.f1_scores_window.append(np.mean(self.f1_scores[-self.window_size:]))
