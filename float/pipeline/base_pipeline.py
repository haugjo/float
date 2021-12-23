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
from numpy.random import Generator
import sys
from tabulate import tabulate
import time
import tracemalloc
from tracemalloc import Snapshot
from typing import Optional, Union, List, Tuple
import warnings

from float.data.data_loader import DataLoader
from float.feature_selection import BaseFeatureSelector
from float.feature_selection.evaluation import FeatureSelectionEvaluator
from float.change_detection import BaseChangeDetector
from float.change_detection.evaluation import ChangeDetectionEvaluator
from float.change_detection.river import RiverChangeDetector
from float.change_detection.skmultiflow import SkmultiflowChangeDetector
from float import pipeline
from float.prediction import BasePredictor
from float.prediction.evaluation import PredictionEvaluator
from float.prediction.river import RiverClassifier
from float.prediction.skmultiflow import SkmultiflowClassifier


class BasePipeline(metaclass=ABCMeta):
    """Abstract base class for evaluation pipelines.

    Attributes:
        data_loader (DataLoader): Data loader object.
        predictors (List[BasePredictor] | None): Predictive model.
        prediction_evaluators (List[PredictionEvaluator] | None): Evaluator for predictive model.
        change_detector (ConceptDriftDetector | None): Concept drift detection model.
        change_detection_evaluator (ChangeDetectionEvaluator | None): Evaluator for active concept drift detection.
        feature_selector (BaseFeatureSelector | None): Online feature selection model.
        feature_selection_evaluator (FeatureSelectionEvaluator | None): Evaluator for the online feature selection.
        batch_size (int | None): Batch size, i.e. no. of observations drawn from the data loader at one time step.
        n_pretrain (int | None): Number of observations used for the initial training of the predictive model.
        n_max (int | None): Maximum number of observations used in the evaluation.
        label_delay_range:
            The min and max delay in the availability of labels in time steps. The delay is sampled uniformly from
            this range.
        known_drifts (List[int] | List[tuple] | None):
            The positions in the dataset (indices) corresponding to known concept drifts.
        estimate_memory_alloc (bool):
            Boolean that indicates if the method-wise change in allocated memory (GB) shall be monitored.
            Note that this delivers only an indication of the approximate memory consumption and can significantly
            increase the total run time of the pipeline.
        test_interval (int):
            The interval/frequency at which the online learning models are evaluated. This parameter is always 1 for a
            prequential evaluation.
        rng (Generator): A numpy random number generator object.
        start_time (float): Physical start time.
        time_step (int): Current logical time step, i.e. iteration.
        n_total (int): Total number of observations currently observed.
    """

    def __init__(self,
                 data_loader: DataLoader,
                 predictors: Optional[List[BasePredictor]],
                 prediction_evaluators: Optional[List[PredictionEvaluator]],
                 change_detector: Optional[BaseChangeDetector],
                 change_detection_evaluator: Optional[ChangeDetectionEvaluator],
                 feature_selector: Optional[BaseFeatureSelector],
                 feature_selection_evaluator: Optional[FeatureSelectionEvaluator],
                 batch_size: int,
                 n_pretrain: int,
                 n_max: int,
                 label_delay_range: Optional[tuple],
                 known_drifts: Optional[Union[List[int], List[tuple]]],
                 estimate_memory_alloc: bool,
                 test_interval: int,
                 random_state: int):
        """Initializes the pipeline.

        Args:
            data_loader: Data loader object.
            predictors: Predictive model(s).
            prediction_evaluators: Evaluator(s) for predictive model(s).
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
            test_interval:
                The interval/frequency at which the online learning models are evaluated. This parameter is always 1 for
                a prequential evaluation.
            random_state: A random integer seed used to specify a random number generator.

        Raises:
            AttributeError: If one of the provided objects is not valid.
        """
        self.data_loader = data_loader
        self.predictors = predictors
        self.prediction_evaluators = prediction_evaluators
        self.change_detector = change_detector
        self.change_detection_evaluator = change_detection_evaluator
        self.feature_selector = feature_selector
        self.feature_selection_evaluator = feature_selection_evaluator
        self.batch_size = batch_size
        self.n_pretrain = n_pretrain
        self.n_max = n_max
        self.label_delay_range = label_delay_range
        self.known_drifts = known_drifts
        self.estimate_memory_alloc = estimate_memory_alloc
        self.test_interval = test_interval
        self.rng = np.random.default_rng(seed=random_state)

        if self.label_delay_range:
            self.sample_buffer = list()

        self.start_time = 0
        self.time_step = 0
        self.n_total = 0

        self._validate()

    @abstractmethod
    def run(self):
        """Runs the pipeline."""
        raise NotImplementedError

    def _validate(self):
        """Validates the input parameters.

        Raises:
            AttributeError: If a crucial parameter to run the pipeline is missing or is invalid.
        """
        if type(self.data_loader) is not DataLoader:
            raise AttributeError('No valid DataLoader object was provided.')

        if not issubclass(type(self.feature_selector), BaseFeatureSelector) and \
                not issubclass(type(self.change_detector), BaseChangeDetector) and \
                not (any(issubclass(type(predictor), BasePredictor) for predictor in self.predictors)):
            raise AttributeError('No valid FeatureSelector, ChangeDetector or Predictor object was provided.')

        if self.predictors and len(self.predictors) != len(self.prediction_evaluators):
            raise AttributeError('A PredictionEvaluator object needs to be provided for every Predictor object.')

        for i in range(len(self.predictors)):
            if type(self.predictors[i]) is RiverClassifier:
                if not self.predictors[i].can_mini_batch:
                    warnings.warn('This classifier does not support batch processing. The batch size is set to 1 for all '
                                  'predictors regardless of the specified batch size.')
                    self.batch_size = 1

        if self.change_detector:
            if self.change_detector.error_based and \
                    not any(issubclass(type(predictor), BasePredictor) for predictor in self.predictors):
                raise AttributeError('An error-based Change Detector cannot be used without a valid Predictor '
                                     'object.')

            if not self.change_detection_evaluator:
                raise AttributeError('A ChangeDetectionEvaluator object needs to be provided when a ChangeDetector 7'
                                     'object is provided.')

        if self.feature_selector:
            if not self.feature_selector.supports_multi_class and self.data_loader.stream.n_classes > 2:
                raise AttributeError('The provided Feature Selector does not support multiclass targets.')

            if not self.feature_selection_evaluator:
                raise AttributeError('A FeatureSelectionEvaluator object needs to be provided when a FeatureSelector '
                                     'object is provided.')

        if len(self.predictors) == 1 and type(self) is pipeline.DistributedFoldPipeline:
            raise AttributeError('The DistributedFoldPipeline can only be used with more than one predictor object.')

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

        if self.predictors:
            for i in range(len(self.predictors)):
                X, y = self.data_loader.get_data(self.n_pretrain)

                if type(self.predictors[i]) is RiverClassifier:
                    if not self.predictors[i].can_mini_batch:
                        for x, y in zip(X, y):
                            self.predictors[i].partial_fit(X=x, y=y)
                        self.n_total += self.n_pretrain
                        return

                self.predictors[i].partial_fit(X=copy.copy(X), y=copy.copy(y))
                self.n_total += self.n_pretrain

    def _run_iteration(self,
                       train_set: Tuple[ArrayLike, ArrayLike],
                       test_set: Optional[Tuple[ArrayLike, ArrayLike]] = None,
                       predictor_train_idx: Optional[List[int]] = None,
                       predictor_test_idx: Optional[List[int]] = None,
                       train_weights: Optional[List[int]] = None,
                       last_iteration: bool = False):
        """Runs an evaluation iteration.

        Args:
            train_set: The observations and labels used for training in the current iteration.
            test_set: The observations and labels used for testing in the current iteration.
            predictor_train_idx:
                (only used for DistributedFoldPipeline) The indices for which predictors should be
                used for training in this iteration.
            predictor_test_idx:
                (only used for DistributedFoldPipeline) The indices for which predictors should be
                used for testing in this iterations
            train_weights:
                (only used for DistributedFoldPipeline in bootstrap mode) The weights that determine
                how much the current training sample/batch should be weighted for each predictor.
            last_iteration (bool): True if this is the last evaluation iteration, False otherwise.
        """
        X_train, y_train = train_set
        X_test, y_test = test_set if test_set else train_set

        predictor_train_idx = predictor_train_idx if predictor_train_idx is not None else [0]
        predictor_test_idx = predictor_test_idx if predictor_test_idx is not None else [0]

        # ----------------------------------------
        # Online Feature Selection
        # ----------------------------------------
        if self.feature_selector:
            if self.estimate_memory_alloc:
                start_snapshot = tracemalloc.take_snapshot()

            start_time = time.time()
            self.feature_selector.weight_features(X=copy.copy(X_train), y=copy.copy(y_train))
            self.feature_selection_evaluator.comp_times.append(time.time() - start_time)

            if self.estimate_memory_alloc:
                self.feature_selection_evaluator.memory_changes.append(
                    self._get_memory_snapshot_diff(start_snapshot=start_snapshot))

            X_train = self.feature_selector.select_features(X=copy.copy(X_train), rng=self.rng)

            if not self.time_step % self.test_interval:
                self.feature_selection_evaluator.run(self.feature_selector.selected_features_history,
                                                     self.feature_selector.n_total_features)

        # ----------------------------------------
        # Prediction
        # ----------------------------------------
        y_pred = None
        if self.predictors:
            if (self.n_pretrain > 0 or self.time_step > 0) and X_test.shape[0] > 0:  # Predict/Test if model has already been trained.
                start_time = time.time()
                for i in predictor_test_idx:
                    y_pred = self.predictors[i].predict(X_test)
                    self.prediction_evaluators[i].testing_comp_times.append(time.time() - start_time)

                    if not self.time_step % self.test_interval:
                        self.prediction_evaluators[i].run(y_true=copy.copy(y_test),
                                                          y_pred=copy.copy(y_pred),
                                                          X=copy.copy(X_test),
                                                          predictor=self.predictors[i],
                                                          rng=self.rng)

            for i, idx in enumerate(predictor_train_idx):
                if self.estimate_memory_alloc:
                    start_snapshot = tracemalloc.take_snapshot()

                start_time = time.time()
                X_train_weighted = X_train.copy()
                y_train_weighted = y_train.copy()
                if train_weights:
                    X_train_weighted, y_train_weighted = np.repeat(X_train, train_weights[i], axis=0), np.repeat(y_train, train_weights[i])

                self.predictors[idx].partial_fit(X_train_weighted, y_train_weighted)
                self.prediction_evaluators[idx].training_comp_times.append(time.time() - start_time)

                if self.estimate_memory_alloc:
                    self.prediction_evaluators[idx].memory_changes.append(
                        self._get_memory_snapshot_diff(start_snapshot=start_snapshot))

        # ----------------------------------------
        # Concept Drift Detection
        # ----------------------------------------
        if self.change_detector:
            if self.estimate_memory_alloc:
                start_snapshot = tracemalloc.take_snapshot()

            start_time = time.time()
            if self.change_detector.error_based:
                if y_pred is not None:
                    # If the predictor has not been pre-trained, then there is no prediction in the first time step.
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
                self.change_detector.drifts.append(self.time_step)

                # Reset modules
                if self.data_loader.scaler:
                    if self.data_loader.scaler.reset_after_drift:
                        self.data_loader.scaler.reset()
                if self.feature_selector:
                    if self.feature_selector.reset_after_drift:
                        self.feature_selector.reset()
                if self.predictors:
                    for i in range(len(self.predictors)):
                        if self.predictors[i].reset_after_drift:
                            self.predictors[i].reset(X=copy.copy(X_train), y=copy.copy(y_train))
                if self.change_detector:
                    if self.change_detector.reset_after_drift:
                        self.change_detector.reset()

            partial_change_detected, partial_change_features = self.change_detector.detect_partial_change()
            if partial_change_detected:
                self.change_detector.partial_drifts.append((self.time_step, partial_change_features))

            if last_iteration:  # The concept drift detection is only evaluated in the last iteration.
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

    def _get_train_set(self, n_batch: int) -> Tuple[ArrayLike, ArrayLike]:
        """Returns the training set to be used for the current iteration.

        Args:
            n_batch: the batch size

        Returns:
            Tuple[ArrayLike, ArrayLike]: the samples and their labels to be used for training
        """
        X, y = self.data_loader.get_data(n_batch=n_batch)
        if self.label_delay_range:
            self.sample_buffer.extend(list(zip(X, y, self.time_step + np.random.randint(self.label_delay_range[0], self.label_delay_range[1], X.shape[0]))))
            if self.time_step >= self.label_delay_range[1]:
                train_set = (np.array([X for (X, _, time_step) in self.sample_buffer if time_step <= self.time_step]), np.array([y for (_, y, time_step) in self.sample_buffer if time_step <= self.time_step]))
                self.sample_buffer = [(X, y, time_step) for (X, y, time_step) in self.sample_buffer if self.time_step < time_step]
            else:
                train_set = (X, y)
        else:
            train_set = (X, y)
        return train_set

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
        print(f'Data Set {self.data_loader.path}')
        print('The pipeline has processed {} instances in total, using batches of size {}.'.format(self.n_total, self.batch_size))

        if self.feature_selector:
            print('----------------------')
            print('Feature Selection ({}/{} features):'.format(self.feature_selector.n_selected_features,
                                                               self.feature_selector.n_total_features))
            print(tabulate({
                **{'Model': [type(self.feature_selector).__name__.split('.')[-1]],
                   'Avg. Comp. Time': [np.mean(self.feature_selection_evaluator.comp_times)]},
                **{'Avg. ' + key: [value['mean'][-1]] for key, value in self.feature_selection_evaluator.result.items()}
            }, headers="keys", tablefmt='github'))

        if self.change_detector:
            print('----------------------')
            print('Concept Drift Detection:')
            print(tabulate({
                **{'Model': [type(self.change_detector).__name__.split('.')[-1] + '.' + type(self.change_detector.detector).__name__
                             if type(self.change_detector) in [SkmultiflowChangeDetector, RiverChangeDetector]
                             else type(self.change_detector).__name__.split('.')[-1]],
                   'Avg. Comp. Time': [np.mean(self.change_detection_evaluator.comp_times)],
                   'Detected Global Drifts': [self.change_detector.drifts] if len(
                       self.change_detector.drifts) <= 5 else [
                       str(self.change_detector.drifts[:5])[:-1] + ', ...]']},
                **{'Avg. ' + key: [np.mean([x for x in value if x is not None]) if len([x for x in value if x is not None]) > 0 else 'N/A']
                if type(value) is list else [value['mean']] for key, value in self.change_detection_evaluator.result.items()}
            }, headers="keys", tablefmt='github'))

        if self.predictors:
            for i in range(len(self.predictors)):
                print('----------------------')
                print('Prediction:')
                if len(self.predictors) > 1:
                    print(f'Predictor {i}:')
                print(tabulate({
                    **{'Model': [type(self.predictors[i]).__name__.split('.')[-1] + '.' + type(self.predictors[i].model).__name__
                                 if type(self.predictors[i]) in [SkmultiflowClassifier, RiverClassifier] else
                                 type(self.predictors[i]).__name__.split('.')[-1]],
                       'Avg. Test Comp. Time': [np.mean(self.prediction_evaluators[i].testing_comp_times)] if self.prediction_evaluators[i].testing_comp_times else ['N/A'],
                       'Avg. Train Comp. Time': [np.mean(self.prediction_evaluators[i].training_comp_times)] if self.prediction_evaluators[i].training_comp_times else ['N/A']},
                    **{'Avg. ' + key: [value['mean'][-1]] if value['mean'] else ['N/A'] for key, value in self.prediction_evaluators[i].result.items()}
                }, headers="keys", tablefmt='github'))
        print('#############################################################################')
