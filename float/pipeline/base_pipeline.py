"""Base Pipeline Module.

This module contains functionality to construct a pipeline and run experiments in a standardized and modular fashion.
In general, we recommend building custom experiments around a pipeline object. This abstract BasePipeline class should
be used as a super class for all specific evaluation pipelines.

Copyright (C) 2021 Johannes Haug

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from abc import ABCMeta, abstractmethod
import copy
import math
import numpy as np
from numpy.typing import ArrayLike
import sys
from tabulate import tabulate
import time
import traceback
import tracemalloc
from tracemalloc import Snapshot
from typing import Optional, Union, List, Tuple

from float.data.data_loader import DataLoader
from float.feature_selection import BaseFeatureSelector
from float.feature_selection.evaluation import FeatureSelectionEvaluator
from float.change_detection import BaseChangeDetector
from float.change_detection.skmultiflow import SkmultiflowChangeDetector
from float.change_detection.evaluation import ChangeDetectionEvaluator
from float.prediction import BasePredictor
from float.prediction.evaluation import PredictionEvaluator
from float.prediction.skmultiflow import SkmultiflowClassifier


class BasePipeline(metaclass=ABCMeta):
    """Abstract base class for evaluation pipelines.

    Attributes:
        data_loader (DataLoader): Data loader object.
        predictor (BasePredictor | None): Predictive model.
        prediction_evaluator (PredictionEvaluator | None): Evaluator for predictive model.
        change_detector (ConceptDriftDetector | None): Concept drift detection model.
        change_detection_evaluator (ChangeDetectionEvaluator | None): Evaluator for active concept drift detection.
        feature_selector (BaseFeatureSelector | None): Online feature selection model.
        feature_selection_evaluator (FeatureSelectionEvaluator | None): Evaluator for the online feature selection.
        batch_size (int | None): Batch size, i.e. no. of observations drawn from the data loader at one time step.
        n_pretrain (int | None): Number of observations used for the initial training of the predictive model.
        n_max (int | None): Maximum number of observations used in the evaluation.
        known_drifts (List[int] | List[tuple] | None):
            The positions in the dataset (indices) corresponding to known concept drifts.
        estimate_memory_alloc (bool):
            Boolean that indicates if the method-wise change in allocated memory (GB) shall be monitored.
            Note that this delivers only an indication of the approximate memory consumption and can significantly
            increase the total run time of the pipeline.
        evaluation_interval (int | None):
            The interval/frequency at which the online learning models are evaluated. This parameter is only relevant in
            a periodic Holdout evaluation.
        start_time (float): Physical start time.
        time_step (int): Current logical time step, i.e. iteration.
        n_total (int): Total number of observations currently observed.
    """
    def __init__(self, data_loader: DataLoader,
                 predictor: Optional[BasePredictor] = None,
                 prediction_evaluator: Optional[PredictionEvaluator] = None,
                 change_detector: Optional[BaseChangeDetector] = None,
                 change_detection_evaluator: Optional[ChangeDetectionEvaluator] = None,
                 feature_selector: Optional[BaseFeatureSelector] = None,
                 feature_selection_evaluator: Optional[FeatureSelectionEvaluator] = None,
                 batch_size: int = 1,
                 n_pretrain: int = 100, n_max: int = np.inf,
                 known_drifts: Optional[Union[List[int], List[tuple]]] = None,
                 estimate_memory_alloc: bool = False,
                 evaluation_interval: Optional[int] = None):
        """Initializes the pipeline.

        Args:
            data_loader: Data loader object.
            predictor: Predictive model.
            prediction_evaluator: Evaluator for predictive model.
            change_detector: Concept drift detection model.
            change_detection_evaluator: Evaluator for active concept drift detection.
            feature_selector: Online feature selection model.
            feature_selection_evaluator: Evaluator for the online feature selection.
            batch_size: Batch size, i.e. no. of observations drawn from the data loader at one time step.
            n_pretrain: Number of observations used for the initial training of the predictive model.
            n_max: Maximum number of observations used in the evaluation.
            known_drifts: The positions in the dataset (indices) corresponding to known concept drifts.
            estimate_memory_alloc:
                Boolean that indicates if the method-wise change in allocated memory (GB) shall be monitored.
                Note that this delivers only an indication of the approximate memory consumption and can significantly
                increase the total run time of the pipeline.
            evaluation_interval: Todo: remove from abstract class, as it is only used by the Holdout pipeline
                The interval/frequency at which the online learning models are evaluated. This parameter is only
                relevant in a periodic Holdout evaluation.

        Raises:
            AttributeError: If one of the provided objects is not valid.
        """
        self.data_loader = data_loader
        self.predictor = predictor
        self.prediction_evaluator = prediction_evaluator
        self.change_detector = change_detector
        self.change_detection_evaluator = change_detection_evaluator
        self.feature_selector = feature_selector
        self.feature_selection_evaluator = feature_selection_evaluator
        self.batch_size = batch_size
        self.n_pretrain = n_pretrain
        self.n_max = n_max
        self.known_drifts = known_drifts
        self.estimate_memory_alloc = estimate_memory_alloc
        self.evaluation_interval = evaluation_interval if evaluation_interval else 1

        self.start_time = 0
        self.time_step = 0
        self.n_total = 0

        try:
            self._validate()
        except AttributeError:
            traceback.print_exc(limit=1)
            return

    @abstractmethod
    def run(self):
        """Runs the pipeline."""
        raise NotImplementedError

    def _validate(self):
        """Validates the input parameters.

        Raises:
            AttributeError: If a crucial parameter to run the pipeline is missing.
        """
        if type(self.data_loader) is not DataLoader:
            raise AttributeError("No valid DataLoader object was provided.")
        if not issubclass(type(self.feature_selector), BaseFeatureSelector) and \
                not issubclass(type(self.change_detector), BaseChangeDetector) and \
                not issubclass(type(self.predictor), BasePredictor):
            raise AttributeError("No valid FeatureSelector, ConceptDriftDetector or Predictor object was provided.")
        if self.change_detector:
            if self.change_detector.error_based and not issubclass(type(self.predictor), BasePredictor):
                raise AttributeError("An error-based Concept Drift Detector cannot be used without a valid Predictor "
                                     "object.")

    def _start_evaluation(self):
        """Starts the evaluation."""
        if self.estimate_memory_alloc:
            tracemalloc.start()

        self.start_time = time.time()
        if self.n_pretrain > 0:
            self._pretrain_predictor()

    def _pretrain_predictor(self):
        """Pretrains the predictive model."""
        print("Pretrain the predictor with {} observation(s).".format(self.n_pretrain))

        if self.predictor:
            X, y = self.data_loader.get_data(self.n_pretrain)

            self.predictor.partial_fit(X=copy.copy(X), y=copy.copy(y))
            self.n_total += self.n_pretrain

    def _run_iteration(self, train_set: Tuple[ArrayLike, ArrayLike],
                       test_set: Optional[Tuple[ArrayLike, ArrayLike]] = None, last_iteration: bool = False):
        """Runs an evaluation iteration.

        Args:
            train_set: The observations and labels used for training in the current iteration.
            test_set: The observations and labels used for testing in the current iteration.
            last_iteration (bool): True if this is the last evaluation iteration, False otherwise.
        """
        X_train, y_train = train_set
        X_test, y_test = test_set if test_set else train_set

        if self.feature_selector:
            if self.estimate_memory_alloc:
                start_snapshot = tracemalloc.take_snapshot()

            start_time = time.time()
            self.feature_selector.weight_features(X=copy.copy(X_train), y=copy.copy(y_train))
            X_train = self.feature_selector.select_features(X=copy.copy(X_train))
            self.feature_selection_evaluator.comp_times.append(time.time() - start_time)

            if self.estimate_memory_alloc:
                self.feature_selection_evaluator.memory_changes.append(
                    self._get_memory_snapshot_diff(start_snapshot=start_snapshot))

            self.feature_selection_evaluator.run(self.feature_selector.selected_features,
                                                 self.feature_selector.n_total_features)

        if self.predictor:
            start_time = time.time()
            y_pred = self.predictor.predict(X_test)
            self.prediction_evaluator.testing_comp_times.append(time.time() - start_time)

            if not self.time_step == 0 and not self.time_step % self.evaluation_interval:  # Todo: why not evaluate at time step t=0?
                self.prediction_evaluator.run(y_true=copy.copy(y_test), y_pred=copy.copy(y_pred), X=copy.copy(X_test),
                                              predictor=self.predictor)

            if self.estimate_memory_alloc:
                start_snapshot = tracemalloc.take_snapshot()

            start_time = time.time()
            self.predictor.partial_fit(X_train, y_train)
            self.prediction_evaluator.training_comp_times.append(time.time() - start_time)

            if self.estimate_memory_alloc:
                self.prediction_evaluator.memory_changes.append(
                    self._get_memory_snapshot_diff(start_snapshot=start_snapshot))

        if self.change_detector:
            if self.estimate_memory_alloc:
                start_snapshot = tracemalloc.take_snapshot()

            start_time = time.time()
            if self.change_detector.error_based:
                for val in (y_pred == y_test):
                    self.change_detector.partial_fit(val)
            else:
                self.change_detector.partial_fit(X=copy.copy(X_train), y=copy.copy(y_train))
            self.change_detection_evaluator.comp_times.append(time.time() - start_time)

            if self.estimate_memory_alloc:
                self.change_detection_evaluator.memory_changes.append(
                    self._get_memory_snapshot_diff(start_snapshot=start_snapshot))

            if self.change_detector.detect_warning_zone():
                self.change_detector.warnings.append(self.time_step)

            if self.change_detector.detect_change():
                if self.time_step not in self.change_detector.drifts:  # Todo: is this if-clause really necessary?
                    self.change_detector.drifts.append(self.time_step)

                # Reset modules
                if self.data_loader.scaler:
                    if self.data_loader.scaler.reset_after_drift:
                        self.data_loader.scaler.reset()
                if self.feature_selector:
                    if self.feature_selector.reset_after_drift:
                        self.feature_selector.reset()
                if self.predictor:
                    if self.predictor.reset_after_drift:
                        self.predictor.reset(X=copy.copy(X_train), y=copy.copy(y_train))
                if self.change_detector:
                    if self.change_detector.reset_after_drift:
                        self.change_detector.reset()

            partial_change_detected, partial_change_features = self.change_detector.detect_partial_change()
            if partial_change_detected:
                if self.time_step not in self.change_detector.partial_drifts:  # Todo: is this if-clause really necessary?
                    self.change_detector.partial_drifts.append((self.time_step, partial_change_features))

            if last_iteration:
                self.change_detection_evaluator.run(self.change_detector.drifts)

    def _get_n_batch(self) -> int:
        """Returns the number of observations that need to be drawn in this iteration.

        Returns:
            int: Batch size.
        """
        if self.n_total + self.batch_size <= self.n_max:
            n_batch = self.batch_size
        else:
            n_batch = self.n_max - self.n_total
        return n_batch

    @staticmethod
    def _get_memory_snapshot_diff(start_snapshot: Snapshot) -> float:
        """Returns the absolute different in allocated memory between two snapshots.

        Args:
            start_snapshot: The initial memory snapshot that was obtained by tracemalloc.

        Returns:
            float: Difference of allocated memory in GB
        """
        end_snapshot = tracemalloc.take_snapshot()
        differences = end_snapshot.compare_to(old_snapshot=start_snapshot, key_type='lineno', cumulative=True)
        return np.sum([stat.size_diff for stat in differences]) * 0.000000001

    def _finish_iteration(self, n_batch: int):
        """Finishes one training iteration, i.e. time step.

        Args:
            n_batch:
                Number of observation that were processed in the iteration. This equals the batch size in all but the
                last iteration.
        """
        self.time_step += 1
        self.n_total += n_batch
        self._update_progress_bar()

    def _update_progress_bar(self):
        """Updates the progress bar in the console."""
        progress = math.ceil(self.n_total / self.n_max * 100)

        if self.change_detector:
            n_detections = len(self.change_detector.drifts)
            last_drift = self.change_detector.drifts[-1] if n_detections > 0 else 0
            out_text = "[%-20s] %d%%, No. of detected drifts: %d, Last detected drift at t=%d." % ('=' * int(0.2 * progress), progress, n_detections, last_drift)
        else:
            out_text = "[%-20s] %d%%" % ('=' * int(0.2 * progress), progress)

        sys.stdout.write('\r')
        sys.stdout.write(out_text)
        sys.stdout.flush()

    def _finish_evaluation(self):
        """Finishes the evaluation."""
        if self.estimate_memory_alloc:
            tracemalloc.stop()

        self.data_loader.stream.restart()
        self._print_summary()

    def _print_summary(self):
        """Prints a summary of the evaluation to the console."""
        print('\n################################## SUMMARY ##################################')
        print('Evaluation has finished after {}s'.format(time.time() - self.start_time))
        print(f'Data Set {self.data_loader.file_path}')
        print('The pipeline has processed {} instances in total, using batches of size {}.'.format(self.n_total, self.batch_size))

        if self.feature_selector:
            print('----------------------')
            print('Feature Selection ({}/{} features):'.format(self.feature_selector.n_selected_features,
                                                               self.feature_selector.n_total_features))
            print(tabulate({
                **{'Model': [type(self.feature_selector).__name__.split('.')[-1]],
                   'Avg. Comp. Time': [np.mean(self.feature_selection_evaluator.comp_times)],
                   'Avg. Change of RAM (GB)': [np.mean(self.feature_selection_evaluator.memory_changes)]},
                **{'Avg. ' + key: [value['mean'][-1]] for key, value in self.feature_selection_evaluator.result.items()}
            }
                , headers="keys", tablefmt='github'))

        if self.change_detector:
            print('----------------------')
            print('Concept Drift Detection:')
            print(tabulate({
                **{'Model': [type(self.change_detector.detector).__name__ if type(
                    self.change_detector) is SkmultiflowChangeDetector else
                             type(self.change_detector).__name__.split('.')[-1]],
                   'Avg. Comp. Time': [np.mean(self.change_detection_evaluator.comp_times)],
                   'Avg. Change of RAM (GB)': [np.mean(self.change_detection_evaluator.memory_changes)],
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
                **{'Model': [type(self.predictor.model).__name__ if type(
                    self.predictor) is SkmultiflowClassifier else
                             type(self.predictor).__name__.split('.')[-1]],
                   'Avg. Test Comp. Time': [np.mean(self.prediction_evaluator.testing_comp_times)],
                   'Avg. Train Comp. Time': [np.mean(self.prediction_evaluator.training_comp_times)],
                   'Avg. Change of RAM (GB)': [np.mean(self.prediction_evaluator.memory_changes)]},
                **{'Avg. ' + key: [value['mean'][-1]] for key, value in self.prediction_evaluator.result.items()}
            }, headers="keys", tablefmt='github'))
        print('#############################################################################')
