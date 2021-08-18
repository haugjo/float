from abc import ABCMeta, abstractmethod
import numpy as np
import time
import copy
import sys
from tabulate import tabulate
from float.data.data_loader import DataLoader
from float.feature_selection import FeatureSelector
from float.concept_drift_detection import ConceptDriftDetector, SkmultiflowDriftDetector
from float.prediction import Predictor


class Pipeline(metaclass=ABCMeta):
    """
    Abstract base class which triggers events for different kinds of training procedures.
    """

    def __init__(self, data_loader, feature_selector, concept_drift_detector, predictor, max_n_samples,
                 batch_size, n_pretrain_samples, known_drifts):
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
            known_drifts (list): list of known concept drifts for this stream
        """
        self.data_loader = data_loader
        self.feature_selector = feature_selector
        self.concept_drift_detector = concept_drift_detector
        self.predictor = predictor

        self.max_n_samples = max_n_samples
        self.batch_size = batch_size
        self.n_pretrain_samples = n_pretrain_samples

        self.known_drifts = known_drifts

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
        if not issubclass(type(self.feature_selector), FeatureSelector) and \
                not issubclass(type(self.concept_drift_detector), ConceptDriftDetector) and \
                not issubclass(type(self.predictor), Predictor):
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
        last_iteration = False

        if self.n_global_samples + self.batch_size <= self.max_n_samples:
            n_samples = self.batch_size
        else:
            n_samples = self.max_n_samples - self.n_global_samples

        if self.n_global_samples + n_samples >= self.max_n_samples:
            last_iteration = True

        X, y = self.data_loader.get_data(n_samples)

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

            start_time = time.time()
            self.predictor.partial_fit(X, y)
            self.predictor.training_times.append(time.time() - start_time)

            self.predictor.evaluate(X, y)

        if self.concept_drift_detector:
            start_time = time.time()
            if self.concept_drift_detector.prediction_based:
                for val in (prediction == y):
                    self.concept_drift_detector.partial_fit(val)
            else:
                self.concept_drift_detector.partial_fit(X, y)
            if self.concept_drift_detector.detected_global_change():
                print(f"Global change detected at time step {self.time_step}")
            self.concept_drift_detector.evaluate(self.time_step, self.max_n_samples, self.known_drifts, self.batch_size,
                                                 last_iteration)
            self.concept_drift_detector.comp_times.append(time.time() - start_time)

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
        print(f'Data Set {self.data_loader.file_path}')
        print('Processed {} instances in batches of {}'.format(self.n_global_samples, self.batch_size))

        if self.feature_selector:
            print('----------------------')
            print('Feature Selection ({}/{} features):'.format(self.feature_selector.n_selected_features,
                                                               self.feature_selector.n_total_features))
            print(tabulate({
                **{'Model': [type(self.feature_selector).__name__.split('.')[-1]],
                   'Avg. Time': [np.mean(self.feature_selector.comp_times)]},
                **{'Avg. ' + key: [np.mean(value)] for key, value in self.feature_selector.evaluation.items()}
            }
                , headers="keys", tablefmt='github'))

        if self.concept_drift_detector:
            print('----------------------')
            print('Concept Drift Detection:')
            print(tabulate({
                'Model': [type(self.concept_drift_detector.detector).__name__ if type(
                    self.concept_drift_detector) is SkmultiflowDriftDetector else
                          type(self.concept_drift_detector).__name__.split('.')[-1]],
                'Avg. Time': [np.mean(self.concept_drift_detector.comp_times)],
                'Detected Global Drifts': [self.concept_drift_detector.global_drifts] if len(
                    self.concept_drift_detector.global_drifts) <= 5 else [
                    str(self.concept_drift_detector.global_drifts[:5])[:-1] + ', ...]'],
                'Avg. Delay': [self.concept_drift_detector.average_delay],
                'Avg. True Positive Rate': [np.mean([x[1] for x in self.concept_drift_detector.true_positive_rates])],
                'Avg. False Discovery Rate': [np.mean(
                    [x[1] for x in self.concept_drift_detector.false_discovery_rates if x[1] is not None]) if len(
                    [x[1] for x in self.concept_drift_detector.false_discovery_rates if
                     x[1] is not None]) > 0 else 'N/A'],
                'Avg. Precision': [
                    np.mean([x[1] for x in self.concept_drift_detector.precision_scores if x[1] is not None]) if len(
                        [x[1] for x in self.concept_drift_detector.precision_scores if
                         x[1] is not None]) > 0 else 'N/A']
            }, headers="keys", tablefmt='github'))

        if self.predictor:
            print('----------------------')
            print('Prediction:')
            print(tabulate({
                **{'Model': [type(self.predictor).__name__.split('.')[-1]],
                   'Avg. Test Time': [np.mean(self.predictor.testing_times)],
                   'Avg. Train Time': [np.mean(self.predictor.training_times)]},
                **{'Avg. ' + key: [np.mean(value)] for key, value in self.predictor.evaluation.items()}
            }, headers="keys", tablefmt='github'))
        print('#############################################################################')

    @abstractmethod
    def run(self):
        raise NotImplementedError
