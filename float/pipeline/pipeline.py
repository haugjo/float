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
    def __init__(self, data_loader, feature_selector, concept_drift_detector, predictor, evaluator, max_n_samples,
                 batch_size, n_pretrain_samples, streaming_features):
        """
        Initializes the pipeline.

        Args:
            data_loader (DataLoader): DataLoader object
            feature_selector (FeatureSelector): FeatureSelector object
            concept_drift_detector (ConceptDriftDetector): ConceptDriftDetector object
            predictor (Predictor): Predictor object
            evaluator (Evaluator): Evaluator object
            max_n_samples (int): maximum number of observations used in the evaluation
            batch_size (int): size of one batch (i.e. no. of observations at one time step)
            n_pretrain_samples (int): no. of observations used for initial training of the predictive model
            streaming_features (dict): (time, feature index) tuples to simulate streaming features
        """
        self.data_loader = data_loader
        self.feature_selector = feature_selector
        self.concept_drift_detector = concept_drift_detector
        self.predictor = predictor
        self.evaluator = evaluator

        self.max_n_samples = max_n_samples
        self.batch_size = batch_size
        self.n_pretrain_samples = n_pretrain_samples
        self.streaming_features = streaming_features if streaming_features else dict()

        self.iterator = 1
        self.n_global_samples = 0
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
        if self.n_pretrain_samples > 0:
            self._pretrain_predictor()

    def _finish_iteration(self, n_samples):
        """
        Finishes one iteration routine.

        Args:
            n_samples (int): number of samples in current data batch
        """
        self.iterator += 1
        self.n_global_samples += n_samples

    def _finish_evaluation(self):
        """
        Finishes the evaluation routine.
        """
        self.data_loader.stream.restart()

    def _pretrain_predictor(self):
        """
        Pretrains the predictive model before starting the evaluation.
        """
        print('Pretrain predictor with {} observation(s).'.format(self.n_pretrain_samples))

        X, y = self.data_loader.get_data(self.n_pretrain_samples)

        self.predictor.partial_fit(X=X, y=y)
        self.n_global_samples += self.n_pretrain_samples

    def _run_single_training_iteration(self):
        """
        Executes a single training iteration.
        """
        if self.n_global_samples + self.batch_size <= self.max_n_samples:
            n_samples = self.batch_size
        else:
            n_samples = self.max_n_samples - self.n_global_samples
        X, y = self.data_loader.get_data(n_samples)

        if self.feature_selector:
            if self.feature_selector.supports_streaming_features:
                X = self._simulate_streaming_features(X)

            start_time = time.time()
            self.feature_selector.weight_features(copy.copy(X), copy.copy(y))
            self.feature_selector.comp_time.compute(start_time, time.time())
            self.feature_selector.select_features(X)

            X = self._sparsify_feature_vector(X, self.feature_selector.selection[-1])

        if self.concept_drift_detector:
            # TODO
            pass

        if self.predictor:
            start_time = time.time()
            prediction = self.predictor.predict(X).tolist()
            self.predictor.testing_time.compute(start_time, time.time())
            self.predictor.predictions.append(prediction)

            start_time = time.time()
            self.predictor.partial_fit(X, y)
            self.predictor.training_time.compute(start_time, time.time())

        if self.evaluator:
            # TODO
            pass

        self._finish_iteration(n_samples)

    def _simulate_streaming_features(self, X):
        """
        Simulates streaming features. Removes inactive features as specified in streaming_features.

        Args:
            X (np.ndarray): samples of current batch

        Returns:
            np.ndarray: sparse X
        """
        if self.iterator == 0 and self.iterator not in self.streaming_features:
            self.active_features = np.arange(self.feature_selector.n_total_features)
            warnings.warn(
                'Simulate streaming features: No active features provided at t=0. All features are used instead.')
        elif self.iterator in self.streaming_features:
            self.active_features = self.streaming_features[self.iterator]
            print('New streaming features {} at t={}'.format(self.streaming_features[self.iterator], self.iterator))

        return self._sparsify_feature_vector(X, self.active_features)

    @staticmethod
    def _sparsify_feature_vector(X, active_features):
        """
        'Removes' inactive features from feature vector by setting them to zero.

        Args:
            X (np.ndarray): samples of current batch
            active_features (list): indices of active features

        Returns:
            np.ndarray: sparse feature vector
        """
        sparse_X = np.zeros(X.shape)
        sparse_X[:, active_features] = X[:, active_features]
        return sparse_X

    @abstractmethod
    def run(self):
        raise NotImplementedError
