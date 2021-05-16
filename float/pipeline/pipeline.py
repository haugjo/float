from abc import ABCMeta, abstractmethod
import warnings
import time
import copy
import numpy as np
from float.data.data_loader import DataLoader
from float.feature_selection import FeatureSelector
from float.concept_drift_detection import ConceptDriftDetector
from float.prediction import Predictor
from float.evaluation.evaluator import Evaluator


class Pipeline(metaclass=ABCMeta):
    """
    Abstract base class which triggers events for different kinds of training procedures.
    """
    def __init__(self, data_loader, feature_selector, concept_drift_detector, predictor, evaluator, max_samples,
                 batch_size, pretrain_size, pred_metrics, fs_metrics, streaming_features):
        """
        Initializes the pipeline.

        Args:
            data_loader (DataLoader): DataLoader object
            feature_selector (FeatureSelector): FeatureSelector object
            concept_drift_detector (ConceptDriftDetector): ConceptDriftDetector object
            predictor (Predictor): Predictor object
            evaluator (Evaluator): Evaluator object
            evaluator (Evaluator): Evaluator object
            max_samples (int): maximum number of observations used in the evaluation
            batch_size (int): size of one batch (i.e. no. of observations at one time step)
            pretrain_size (int): no. of observations used for initial training of the predictive model
            pred_metrics (list): predictive metrics/measures
            fs_metrics (list): feature selection metrics/measures
            streaming_features (dict): (time, feature index) tuples to simulate streaming features
        """
        self.data_loader = data_loader
        self.feature_selector = feature_selector
        self.concept_drift_detector = concept_drift_detector
        self.predictor = predictor
        self.evaluator = evaluator

        self.max_samples = max_samples
        self.batch_size = batch_size
        self.pretrain_size = pretrain_size
        self.pred_metrics = pred_metrics
        self.fs_metrics = fs_metrics
        self.streaming_features = dict() if streaming_features is None else streaming_features

        self.iteration = 1
        self.start_time = 0
        self.global_sample_count = 0
        self.active_features = []

        self._check_input()

    def _check_input(self):
        """
        Checks if the provided parameter values are sufficient to run a pipeline.

        Raises:
            AttributeError: if a crucial parameter is missing
        """
        if type(self.data_loader) is not DataLoader:
            raise AttributeError('No valid DataLoader object was provided.')
        if type(self.feature_selector) is not FeatureSelector and \
                type(self.concept_drift_detector) is not ConceptDriftDetector and \
                type(self.predictor) is not Predictor:
            raise AttributeError('No valid FeatureSelector, ConceptDriftDetector or Predictor object was provided.')
        if type(self.evaluator) is not Evaluator:
            warnings.warn('No valid Evaluator object was provided.')

    def _start_evaluation(self):
        """
        Starts the evaluation routine.
        """
        self.start_time = time.time()
        if self.pretrain_size > 0:
            self._pretrain_predictive_model()

    def _finish_iteration(self, samples):
        """
        Finishes one iteration routine.

        Args:
            samples (int): size of current data batch
        """
        self.iteration += 1
        self.global_sample_count += samples

    def _finish_evaluation(self):
        """
        Finishes the evaluation routine.
        """
        self.data_loader.stream.restart()

    def _pretrain_predictive_model(self):
        """
        Pre-trains the predictive model before starting the evaluation.
        """
        print('Pre-train predictor with {} observation(s).'.format(self.pretrain_size))

        X, y = self.data_loader.get_data(self.pretrain_size)

        self.predictor.partial_fit(X=X, y=y)
        self.global_sample_count += self.pretrain_size

    def _one_training_iteration(self):
        """
        Executes one training iteration.
        """
        if self.global_sample_count + self.batch_size <= self.max_samples:
            samples = self.batch_size
        else:
            samples = self.max_samples - self.global_sample_count
        X, y = self.data_loader.get_data(samples)

        if self.feature_selector.supports_streaming_features:
            X = self._simulate_streaming_features(X)

        self.start_time = time.time()
        self.feature_selector.weight_features(copy.copy(X), copy.copy(y))
        self.feature_selector.comp_time.compute(self.start_time, time.time())
        self.feature_selector.select_features(X)
        for metric in self.fs_metrics:
            metric.compute(self.feature_selector)

        X = self._sparsify_X(X, self.feature_selector.selection[-1])

        self.start_time = time.time()
        prediction = self.predictor.predict(X).tolist()
        self.predictor.testing_time.compute(self.start_time, time.time())
        self.predictor.predictions.append(prediction)
        for metric in self.pred_metrics:
            metric.compute(y, prediction)

        self.start_time = time.time()
        self.predictor.partial_fit(X, y)
        self.predictor.training_time.compute(self.start_time, time.time())

        self._finish_iteration(samples)

    def _simulate_streaming_features(self, X):
        """
        Simulates streaming features. Removes inactive features as specified in streaming_features.

        Args:
            X (np.array): samples of current batch

        Returns:
            np.array: sparse X
        """
        if self.iteration == 0 and self.iteration not in self.streaming_features:
            self.active_features = np.arange(self.feature_selector.n_total_ftr)
            warnings.warn(
                'Simulate streaming features: No active features provided at t=0. All features are used instead.')
        elif self.iteration in self.streaming_features:
            self.active_features = self.streaming_features[self.iteration]
            print('New streaming features {} at t={}'.format(self.streaming_features[self.iteration], self.iteration))

        return self._sparsify_X(X, self.active_features)

    @staticmethod
    def _sparsify_X(X, active_features):
        """
        'Removes' inactive features from X by setting them to zero.

        Args:
            X (np.array): samples of current batch
            active_features (list): indices of active features

        Returns:
            np.array: sparse X
        """
        sparse_X = np.zeros(X.shape)
        sparse_X[:, active_features] = X[:, active_features]
        return sparse_X

    @abstractmethod
    def run(self):
        raise NotImplementedError
