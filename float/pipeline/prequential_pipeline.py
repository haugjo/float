import warnings
import time
import copy
import numpy as np
from float.pipeline.pipeline import Pipeline
from float.data.data_loader import DataLoader
from float.feature_selection import FeatureSelector
from float.concept_drift_detection import ConceptDriftDetector
from float.prediction import Predictor
from float.evaluation.evaluator import Evaluator


class PrequentialPipeline(Pipeline):
    """
    Pipeline which implements the test-then-train evaluation.
    TODO: create a structure to outsource code that is shared between child pipelines to parent pipeline

     Attributes:
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
        iteration (int): current iteration (logical time step)
        start_time (float): physical time when starting the evaluation
        global_sample_count (int): no. of observations processed so far
        active_features (list): indices of currently active features (for simulating streaming features)
    """
    def __init__(self, data_loader=None, feature_selector=None, concept_drift_detector=None, predictor=None,
                 evaluator=None, max_samples=100000, batch_size=100, pretrain_size=100, pred_metrics=None,
                 fs_metrics=None, streaming_features=None):
        """
        Initializes the pipeline and checks the input parameters.
        TODO: is there better way of managing the large amount of parameters?

        Args:
            data_loader (DataLoader): DataLoader object
            feature_selector (FeatureSelector): FeatureSelector object
            concept_drift_detector (ConceptDriftDetector): ConceptDriftDetector object
            predictor (Predictor): Predictor object
            evaluator (Evaluator): Evaluator object
            max_samples (int): maximum number of observations used in the evaluation
            batch_size (int): size of one batch (i.e. no. of observations at one time step)
            pretrain_size (int): no. of observations used for initial training of the predictive model
            pred_metrics (list): predictive metrics/measures
            fs_metrics (list): feature selection metrics/measures
            streaming_features (dict): (time, feature index) tuples to simulate streaming features
        """
        super().__init__(data_loader, feature_selector, concept_drift_detector, predictor, evaluator)
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

    def run(self):
        """
        Runs the pipeline.
        """
        if (self.data_loader.stream.n_remaining_samples() > 0) and \
                (self.data_loader.stream.n_remaining_samples() < self.max_samples):
            self.max_samples = self.data_loader.stream.n_samples
            warnings.warn('Parameter max_samples exceeds the size of data_loader and will be automatically reset.',
                          stacklevel=2)

        # Start evaluation
        self._start_evaluation()

        self._test_then_train()

        # Finish evaluation
        self._finish_evaluation()

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

    def _test_then_train(self):
        """
        Test-then-train evaluation
        """
        while self.global_sample_count < self.max_samples:
            try:
                self._one_training_iteration()
            except BaseException as exc:
                print(exc)
                break

    def _one_training_iteration(self):
        """
        Executes one training iteration.
        """
        # Load data batch
        if self.global_sample_count + self.batch_size <= self.max_samples:
            samples = self.batch_size
        else:
            samples = self.max_samples - self.global_sample_count  # all remaining samples
        X, y = self.data_loader.get_data(samples)

        # Simulate streaming features
        if self.feature_selector.supports_streaming_features:
            X = self._simulate_streaming_features(X)

        # Feature Selection
        start = time.time()
        self.feature_selector.weight_features(copy.copy(X), copy.copy(y))
        self.feature_selector.comp_time.compute(start, time.time())
        self.feature_selector.select_features(X)
        for metric in self.fs_metrics:
            metric.compute(self.feature_selector)

        # Retain selected features
        X = self._sparsify_X(X, self.feature_selector.selection[-1])

        # Testing
        start = time.time()
        prediction = self.predictor.predict(X).tolist()
        self.predictor.testing_time.compute(start, time.time())
        self.predictor.predictions.append(prediction)
        for metric in self.pred_metrics:
            metric.compute(y, prediction)

        # Training
        start = time.time()
        self.predictor.partial_fit(X, y)
        self.predictor.training_time.compute(start, time.time())

        # Finish iteration
        self._finish_iteration(self, samples)

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

    def _pretrain_predictive_model(self):
        """
        Pre-trains the predictive model before starting the evaluation.
        """
        print('Pre-train predictor with {} observation(s).'.format(self.pretrain_size))

        X, y = self.data_loader.get_data(self.pretrain_size)

        # Fit model and increase sample count
        self.predictor.partial_fit(X=X, y=y)
        self.global_sample_count += self.pretrain_size
