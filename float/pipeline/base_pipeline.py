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
import numpy as np
from numpy.typing import ArrayLike
from numpy.random import Generator
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
from float.pipeline.utils import validate_pipeline_attrs, update_progress_bar, print_evaluation_summary
from float.prediction import BasePredictor
from float.prediction.evaluation import PredictionEvaluator
from float.prediction.river import RiverClassifier


class BasePipeline(metaclass=ABCMeta):
    """Abstract base class for evaluation pipelines.

    Attributes:
        data_loader (DataLoader): Data loader object.
        predictors (BasePredictor | List[BasePredictor] | None): Predictive model(s).
        prediction_evaluators (PredictionEvaluator | List[PredictionEvaluator] | None):
            Evaluator(s) for the predictive model(s).
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
                 predictor: Optional[Union[BasePredictor, List[BasePredictor]]],
                 prediction_evaluator: Optional[PredictionEvaluator],
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
            predictor: Predictive model(s).
            prediction_evaluator: Evaluator for the predictive model(s).
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

        if isinstance(predictor, list):
            self.predictors = predictor
            self.prediction_evaluators = []  # Make copies of the PredictionEvaluator for each predictor object.
            for i in range(len(self.predictors)):
                self.prediction_evaluators[i] = copy.deepcopy(prediction_evaluator)
        else:
            self.predictors = [predictor]
            self.prediction_evaluators = [prediction_evaluator]

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

        validate_pipeline_attrs(pipeline=self)

    def run(self):
        """Runs the pipeline.

        This function is specifically implemented for each evaluation strategy.
        """
        if (self.data_loader.stream.n_remaining_samples() > 0) and \
                (self.data_loader.stream.n_remaining_samples() < self.n_max):
            self.n_max = self.data_loader.stream.n_remaining_samples()
            warnings.warn("Parameter max_n_samples exceeds the size of the data_loader and will be automatically reset "
                          "to {}.".format(self.n_max), stacklevel=2)

        self._start_evaluation()

    def _start_evaluation(self):
        """Starts the evaluation."""
        if self.estimate_memory_alloc:
            tracemalloc.start()

        self.start_time = time.time()
        if self.predictors is not None and self.n_pretrain > 0:
            self._pretrain_predictor()

    def _pretrain_predictor(self):
        """Pretrains the predictive model."""
        print("Pretrain the predictor with {} observation(s).".format(self.n_pretrain))

        for predictor in self.predictors:
            X, y = self.data_loader.get_data(self.n_pretrain)

            if isinstance(predictor, RiverClassifier) and not predictor.can_mini_batch:
                # Some River classifiers do not support batch-processing. In this case, we pretrain iteratively.
                for x_i, y_i in zip(X, y):
                    predictor.partial_fit(X=x_i, y=y_i)
            else:
                predictor.partial_fit(X=copy.copy(X), y=copy.copy(y))
            self.n_total += self.n_pretrain

    def _run_iteration(self,
                       train_set: Tuple[ArrayLike, ArrayLike],
                       test_set: Tuple[ArrayLike, ArrayLike],
                       last_iteration: bool,
                       predictors_for_training: Union[int, List[int]],
                       predictors_for_testing: Union[int, List[int]],
                       predictors_training_weights: Optional[Union[int, List[int]]] = None):
        """Runs an evaluation iteration.

        Args:
            train_set: The observations and labels used for training in the current iteration.
            test_set: The observations and labels used for testing in the current iteration.
            predictors_for_training:
                The indices of the predictors that are trained in this iteration (this argument is only used for the
                DistributedFoldPipeline).
            predictors_for_testing:
                The indices of the predictors that are used for testing in this iteration (this argument is only
                used for the DistributedFoldPipeline).
            predictors_training_weights:
                The weights that determine how much the current training sample/batch should be weighted for each
                predictor that is trained in this iteration (this argument is only used for the
                DistributedFoldPipeline).
            last_iteration (bool): True if this is the last evaluation iteration, False otherwise.
        """
        X_train, y_train = train_set
        X_test, y_test = test_set

        if len(X_train) == 0:  # Todo: find better solution
            warnings.warn('No samples available with which to train. Metrics will be set to None this iteration.')
            self._set_metrics_to_nan(predictors_for_training, predictors_for_testing)
            return

        # ----------------------------------------
        # Online Feature Selection
        # ----------------------------------------
        if self.feature_selector is not None:
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
        if self.predictors is not None:
            for pred_idx, (predictor, prediction_evaluator) in enumerate(zip(self.predictors, self.prediction_evaluators)):
                # Test, if the predictor has already been trained and the test set is not empty.
                if (self.n_pretrain > 0 or self.time_step > 0) and X_test.shape[0] > 0:
                    # Test in the specified frequency (the interval is 1 for all but the HoldoutPipeline)
                    if pred_idx in predictors_for_testing and self.time_step % self.test_interval == 0:
                        start_time = time.time()
                        y_pred = predictor.predict(X_test)
                        prediction_evaluator.testing_comp_times.append(time.time() - start_time)
                        prediction_evaluator.run(y_true=copy.copy(y_test),
                                                 y_pred=copy.copy(y_pred),
                                                 X=copy.copy(X_test),
                                                 predictor=predictor,
                                                 rng=self.rng)

                # Train the predictor.
                if pred_idx in predictors_for_training:
                    if self.estimate_memory_alloc:
                        start_snapshot = tracemalloc.take_snapshot()

                    start_time = time.time()
                    X_train_weighted = X_train.copy()
                    y_train_weighted = y_train.copy()
                    if predictors_training_weights:
                        # Repeat/Weight training observations acc. to the specified weight.
                        X_train_weighted = np.repeat(X_train, predictors_training_weights[pred_idx], axis=0)
                        y_train_weighted = np.repeat(y_train, predictors_training_weights[pred_idx])

                    predictor.partial_fit(X_train_weighted, y_train_weighted)
                    prediction_evaluator.training_comp_times.append(time.time() - start_time)

                    if self.estimate_memory_alloc:
                        prediction_evaluator.memory_changes.append(
                            self._get_memory_snapshot_diff(start_snapshot=start_snapshot))

        # ----------------------------------------
        # Concept Drift Detection
        # ----------------------------------------
        if self.change_detector is not None:
            if self.estimate_memory_alloc:
                start_snapshot = tracemalloc.take_snapshot()

            start_time = time.time()
            if self.change_detector.error_based:
                # Always use the first predictor for predicting y_pred, which is used for change detection.
                y_pred = self.predictors[0].predict(X_test)
                if y_pred is not None:
                    # If the predictor has not been pre-trained, then there is no prediction in the first time step.
                    self.change_detector.partial_fit([y_pred == y_test])
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
                if self.data_loader.scaler is not None and self.data_loader.scaler.reset_after_drift:
                    self.data_loader.scaler.reset()
                if self.feature_selector is not None and self.feature_selector.reset_after_drift:
                    self.feature_selector.reset()
                for predictor in self.predictors:
                    if predictor is not None and predictor.reset_after_drift:
                        predictor.reset(X=copy.copy(X_train), y=copy.copy(y_train))
                if self.change_detector is not None and self.change_detector.reset_after_drift:
                    self.change_detector.reset()

            partial_change_detected, partial_change_features = self.change_detector.detect_partial_change()
            if partial_change_detected:
                self.change_detector.partial_drifts.append((self.time_step, partial_change_features))

            if last_iteration:  # The concept drift detection is only evaluated in the last iteration.
                self.change_detection_evaluator.run(self.change_detector.drifts)

    def _set_metrics_to_nan(self, predictor_test_idx: List[int], predictor_train_idx: List[int]):
        """
        Set all evaluation metrics to np.nan for when there are no samples with which to train. This can
        happen when label_delay_range is set with a small batch size.

        Args:
            predictor_train_idx:
                (only used for DistributedFoldPipeline) The indices for which predictors should be
                used for training in this iteration.
            predictor_test_idx:
                (only used for DistributedFoldPipeline) The indices for which predictors should be
                used for testing in this iterations
        """
        # Set computation times to nan
        self.feature_selection_evaluator.comp_times.append(np.nan)
        for idx in predictor_test_idx:
            self.prediction_evaluators[idx].testing_comp_times.append(np.nan)
        for idx in predictor_train_idx:
            self.prediction_evaluators[idx].training_comp_times.append(np.nan)
        self.change_detection_evaluator.comp_times.append(np.nan)

        # Set evaluation measures to nan
        if not self.time_step % self.test_interval:
            for measure_func in self.feature_selection_evaluator.measure_funcs:
                self.feature_selection_evaluator.result[measure_func.__name__]['measures'].append(np.nan)
                self.feature_selection_evaluator.result[measure_func.__name__]['mean'].append(np.nan)
                self.feature_selection_evaluator.result[measure_func.__name__]['var'].append(np.nan)

            for idx in predictor_test_idx:
                for measure_func in self.prediction_evaluators[idx].measure_funcs:
                    self.prediction_evaluators[idx].result[measure_func.__name__]['measures'].append(np.nan)
                    self.prediction_evaluators[idx].result[measure_func.__name__]['mean'].append(np.nan)
                    self.prediction_evaluators[idx].result[measure_func.__name__]['var'].append(np.nan)

            for measure_func in self.change_detection_evaluator.measure_funcs:
                self.change_detection_evaluator.result[measure_func.__name__]['measures'] = np.nan
                self.change_detection_evaluator.result[measure_func.__name__]['mean'] = np.nan
                self.change_detection_evaluator.result[measure_func.__name__]['var'] = np.nan

        # Set memory changes to nan
        if self.estimate_memory_alloc:
            self.feature_selection_evaluator.memory_changes.append(np.nan)
            for i in range(len(self.prediction_evaluators)):
                self.prediction_evaluators[i].memory_changes.append(np.nan)
            self.change_detection_evaluator.memory_changes.append(np.nan)

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

    def _get_train_set(self, n_batch: int) -> Tuple[ArrayLike, ArrayLike]:  # Todo: revise!!
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
        update_progress_bar(pipeline=self)

    def _finish_evaluation(self):
        """Finishes the evaluation."""
        if self.estimate_memory_alloc:
            tracemalloc.stop()

        self.data_loader.stream.restart()
        print_evaluation_summary(pipeline=self)
