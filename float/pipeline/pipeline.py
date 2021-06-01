from abc import ABCMeta, abstractmethod
import numpy as np
import time
import copy
import sys
from tabulate import tabulate
from float.data.data_loader import DataLoader
from float.feature_selection import FeatureSelector
from float.concept_drift_detection import ConceptDriftDetector
from float.prediction import Predictor


class Pipeline(metaclass=ABCMeta):
    """
    Abstract base class which triggers events for different kinds of training procedures.
    """

    def __init__(self, data_loader, feature_selector, concept_drift_detector, predictor, max_n_samples,
                 batch_size, n_pretrain_samples):
        """
        Initializes the pipeline.

        Args:
            data_loader (DataLoader): DataLoader object
            feature_selector (FeatureSelector): FeatureSelector object
            concept_drift_detector (ConceptDriftDetector): ConceptDriftDetector object
            predictor (Predictor): Predictor object
            max_n_samples (int): maximum number of observations used in the evaluation
            batch_size (int): size of one batch (i.e. no. of observations at one time step)
            n_pretrain_samples (int): no. of observations used for initial training of the predictive model
        """
        self.data_loader = data_loader
        self.feature_selector = feature_selector
        self.concept_drift_detector = concept_drift_detector
        self.predictor = predictor

        self.max_n_samples = max_n_samples
        self.batch_size = batch_size
        self.n_pretrain_samples = n_pretrain_samples

        self.start_time = 0
        self.time_step = 0
        self.n_global_samples = 0

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

    def _start_evaluation(self):
        """
        Starts the evaluation routine.
        """
        self.start_time = time.time()
        if self.n_pretrain_samples > 0:
            self._pretrain_predictor()

    def _finish_iteration(self, n_samples):
        """
        Finishes one iteration routine.

        Args:
            n_samples (int): number of samples in current data batch
        """
        self.time_step += 1
        self.n_global_samples += n_samples
        self._update_progress_bar()

    def _finish_evaluation(self):
        """
        Finishes the evaluation routine.
        """
        self.data_loader.stream.restart()
        self._print_summary()

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

        if self.concept_drift_detector:
            start_time = time.time()
            self.concept_drift_detector.partial_fit(X)
            self.concept_drift_detector.comp_times.append(time.time() - start_time)
            change_detected = self.concept_drift_detector.detected_global_change()
            self.concept_drift_detector.change_detections.compute(change_detected)
            self.concept_drift_detector.evaluate()
            if change_detected:
                print(f"Change detected at {self.time_step}.")

        if self.feature_selector:
            start_time = time.time()
            self.feature_selector.weight_features(copy.copy(X), copy.copy(y))
            self.feature_selector.comp_times.append(time.time() - start_time)
            X = self.feature_selector.select_features(X, self.time_step)
            self.feature_selector.evaluate()
            if self.feature_selector.supports_streaming_features and \
                    self.time_step in self.feature_selector.streaming_features:
                print('New streaming features {} at t={}'.format(
                    self.feature_selector.streaming_features[self.time_step], self.time_step))

        if self.predictor:
            start_time = time.time()
            prediction = self.predictor.predict(X).tolist()
            self.predictor.testing_times.append(time.time() - start_time)
            self.predictor.predictions.append(prediction)

            start_time = time.time()
            self.predictor.partial_fit(X, y)
            self.predictor.training_times.append(time.time() - start_time)
            self.predictor.evaluate()

        self._finish_iteration(n_samples)

    def _update_progress_bar(self):
        """
        Updates the progress bar.
        """
        j = self.n_global_samples / self.max_n_samples
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
        sys.stdout.flush()

    def _print_summary(self):
        """
        Prints a summary of the evaluation to the console.
        """
        print('\n################################## SUMMARY ##################################')
        print('Evaluation finished after {}s'.format(time.time() - self.start_time))
        print('Processed {} instances in batches of {}'.format(self.n_global_samples, self.batch_size))

        if self.feature_selector:
            print('----------------------')
            print('Feature Selection ({}/{} features):'.format(self.feature_selector.n_selected_features,
                                                               self.feature_selector.n_total_features))
            print(tabulate({
                'Model': [type(self.feature_selector)],
                'Avg. Time': [np.mean(self.feature_selector.comp_times)],
                # 'Avg. {}'.format(self.fs_metrics[0].name): [self.fs_metrics[0].mean] TODO: update
            }, headers="keys", tablefmt='github'))

        if self.concept_drift_detector:
            print('----------------------')
            print('Concept Drift Detection:')
            print(tabulate({
                'Model': [type(self.concept_drift_detector)],
                'Avg. Time': [np.mean(self.concept_drift_detector.comp_times)],
                'Number of Detected Changes:': [self.concept_drift_detector.change_detections.n_changes],
            }, headers="keys", tablefmt='github'))

        if self.predictor:
            print('----------------------')
            print('Prediction:')
            print(tabulate({
                'Model': [type(self.predictor)],
                'Avg. Test Time': [np.mean(self.predictor.testing_times)],
                'Avg. Train Time': [np.mean(self.predictor.training_times)],
                # 'Avg. {}'.format(self.pred_metrics[0].name): [self.pred_metrics[0].mean] TODO: update
            }, headers="keys", tablefmt='github'))
        print('#############################################################################')

    @abstractmethod
    def run(self):
        raise NotImplementedError
