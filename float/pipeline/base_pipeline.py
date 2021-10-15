from abc import ABCMeta, abstractmethod
import numpy as np
import time
import copy
import sys
import traceback
from tabulate import tabulate
from float.data.data_loader import DataLoader
from float.feature_selection import BaseFeatureSelector
from float.feature_selection.evaluation import FeatureSelectionEvaluator
from float.change_detection import BaseChangeDetector
from float.change_detection.skmultiflow import SkmultiflowChangeDetector
from float.change_detection.evaluation import ChangeDetectionEvaluator
from float.prediction import BasePredictor
from float.prediction.evaluation import PredictionEvaluator


class BasePipeline(metaclass=ABCMeta):
    """
    Abstract base class which triggers events for different kinds of training procedures.
    """

    def __init__(self, data_loader, feature_selector, feature_selection_evaluator, change_detector,
                 change_detection_evaluator, predictor, prediction_evaluator, max_n_samples, batch_size,
                 n_pretrain_samples, known_drifts, evaluation_interval=None):
        """
        Initializes the pipeline.

        Args:
            data_loader (DataLoader): DataLoader object
            feature_selector (BaseFeatureSelector | None): FeatureSelector object
            feature_selection_evaluator (FeatureSelectionEvaluator | None): FeatureSelectionEvaluator object
            change_detector (ConceptDriftDetector | None): ConceptDriftDetector object
            change_detection_evaluator (ChangeDetectionEvaluator | None): ChangeDetectionEvaluator object
            predictor (BasePredictor | None): Predictor object
            prediction_evaluator (PredictionEvaluator | None): PredictionEvaluator object
            max_n_samples (int): maximum number of observations used in the evaluation
            batch_size (int): size of one batch (i.e. no. of observations at one time step)
            n_pretrain_samples (int): no. of observations used for initial training of the predictive model
            known_drifts (list): list of known concept drifts for this stream
            evaluation_interval (int): the interval at which the predictor should be evaluated using the test set
        """
        self.data_loader = data_loader
        self.feature_selector = feature_selector
        self.feature_selection_evaluator = feature_selection_evaluator
        self.change_detector = change_detector
        self.change_detection_evaluator = change_detection_evaluator
        self.predictor = predictor
        self.prediction_evaluator = prediction_evaluator

        self.max_n_samples = max_n_samples
        self.batch_size = batch_size
        self.n_pretrain_samples = n_pretrain_samples

        self.known_drifts = known_drifts

        self.evaluation_interval = evaluation_interval if evaluation_interval else 1

        self.start_time = 0
        self.time_step = 0
        self.n_global_samples = 0

        try:
            self.__check_input()
        except AttributeError:
            traceback.print_exc(limit=1)
            return

    def __check_input(self):
        """
        Checks if the provided parameter values are sufficient to run a pipeline.

        Raises:
            AttributeError: if a crucial parameter is missing
        """
        if type(self.data_loader) is not DataLoader:
            raise AttributeError('No valid DataLoader object was provided.')
        if not issubclass(type(self.feature_selector), BaseFeatureSelector) and \
                not issubclass(type(self.change_detector), BaseChangeDetector) and \
                not issubclass(type(self.predictor), BasePredictor):
            raise AttributeError('No valid FeatureSelector, ConceptDriftDetector or Predictor object was provided.')
        if self.change_detector:
            if self.change_detector.error_based and not issubclass(type(self.predictor), BasePredictor):
                raise AttributeError('An error-based Concept Drift Detector cannot be used without a valid Predictor object.')

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

        if self.predictor:
            X, y = self.data_loader.get_data(self.n_pretrain_samples)

            self.predictor.partial_fit(X=X, y=y)
            self.n_global_samples += self.n_pretrain_samples

    def _get_n_samples(self):
        if self.n_global_samples + self.batch_size <= self.max_n_samples:
            n_samples = self.batch_size
        else:
            n_samples = self.max_n_samples - self.n_global_samples
        return n_samples

    def _run_single_training_iteration(self, train_set, test_set=None, last_iteration=False):
        """
        Executes a single training iteration.

        Args:
            train_set (np.ndarray, np.ndarray): the samples and their labels used for training
            test_set ((np.ndarray, np.ndarray) | None): the samples and their labels used for testing
            last_iteration (bool): True if this is the last iteration, False otherwise
        """
        X_train, y_train = train_set
        X_test, y_test = test_set if test_set else train_set

        if self.feature_selector:
            start_time = time.time()
            self.feature_selector.weight_features(copy.copy(X_train), copy.copy(y_train))
            self.feature_selection_evaluator.comp_times.append(time.time() - start_time)

            X_train = self.feature_selector.select_features(X_train)

            self.feature_selection_evaluator.run(self.feature_selector.selected_features, self.feature_selector.n_total_features)

        if self.predictor:
            start_time = time.time()
            y_pred = self.predictor.predict(X_test)
            self.prediction_evaluator.testing_comp_times.append(time.time() - start_time)

            if not self.time_step == 0 and not self.time_step % self.evaluation_interval:  # Todo: why not evaluate at time step t=0?
                self.prediction_evaluator.run(y_test, y_pred, X_test, self.predictor)

            start_time = time.time()
            self.predictor.partial_fit(X_train, y_train)
            self.prediction_evaluator.training_comp_times.append(time.time() - start_time)

        if self.change_detector:
            start_time = time.time()
            if self.change_detector.error_based:
                for val in (y_pred == y_test):  # Todo: make sure that y_pred is available
                    self.change_detector.partial_fit(val)
            else:
                self.change_detector.partial_fit(copy.copy(X_train), copy.copy(y_train))

            if self.change_detector.detect_warning_zone():
                self.change_detector.warnings.append(self.time_step)

            if self.change_detector.detect_change():
                print(f"Global change detected at time step {self.time_step}")
                if self.time_step not in self.change_detector.drifts:  # Todo: is this if-clause really necessary?
                    self.change_detector.drifts.append(self.time_step)

                # Reset modules
                if self.data_loader.scaler.reset_after_drift:
                    self.data_loader.scaler.reset()
                if self.feature_selector.reset_after_drift:
                    self.feature_selector.reset()
                if self.predictor.reset_after_drift:
                    self.predictor.reset(X_train, y_train)
                if self.change_detector.reset_after_drift:
                    self.change_detector.reset()

            partial_change_detected, partial_change_features = self.change_detector.detect_partial_change()
            if partial_change_detected:
                if self.time_step not in self.change_detector.partial_drifts:  # Todo: is this if-clause really necessary?
                    self.change_detector.partial_drifts.append((self.time_step, partial_change_features))

            self.change_detection_evaluator.comp_times.append(time.time() - start_time)
            if last_iteration:
                self.change_detection_evaluator.run(self.change_detector.drifts)

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
        print(f'Data Set {self.data_loader.file_path}')
        print('Processed {} instances in batches of {}'.format(self.n_global_samples, self.batch_size))

        if self.feature_selector:
            print('----------------------')
            print('Feature Selection ({}/{} features):'.format(self.feature_selector.n_selected_features,
                                                               self.feature_selector.n_total_features))
            print(tabulate({
                **{'Model': [type(self.feature_selector).__name__.split('.')[-1]],
                   'Avg. Time': [np.mean(self.feature_selection_evaluator.comp_times)]},
                **{'Avg. ' + key: [value['mean']] for key, value in self.feature_selection_evaluator.result.items()}
            }
                , headers="keys", tablefmt='github'))

        if self.change_detector:
            print('----------------------')
            print('Concept Drift Detection:')
            print(tabulate({
                **{'Model': [type(self.change_detector.detector).__name__ if type(
                    self.change_detector) is SkmultiflowChangeDetector else
                             type(self.change_detector).__name__.split('.')[-1]],
                   'Avg. Time': [np.mean(self.change_detection_evaluator.comp_times)],
                   'Detected Global Drifts': [self.change_detector.drifts] if len(
                       self.change_detector.drifts) <= 5 else [
                       str(self.change_detector.drifts[:5])[:-1] + ', ...]']},
                **{'Avg. ' + key: [np.mean([x for x in value if x is not None]) if len([x for x in value if x is not None]) > 0 else 'N/A']
                if type(value) is list else [value['mean']] for key, value in self.change_detection_evaluator.result.items()}
            }, headers="keys", tablefmt='github'))

        if self.predictor:
            print('----------------------')
            print('Prediction:')
            print(tabulate({
                **{'Model': [type(self.predictor).__name__.split('.')[-1]],
                   'Avg. Test Time': [np.mean(self.prediction_evaluator.testing_comp_times)],
                   'Avg. Train Time': [np.mean(self.prediction_evaluator.training_comp_times)]},
                **{'Avg. ' + key: [value['mean']] for key, value in self.prediction_evaluator.result.items()}
            }, headers="keys", tablefmt='github'))
        print('#############################################################################')

    @abstractmethod
    def run(self):
        raise NotImplementedError
