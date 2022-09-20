"""Base Pipeline.

This module contains functionality to construct a pipeline and run experiments in a standardized and modular fashion.
This abstract BasePipeline class should be used as a super class for all specific evaluation pipelines.

Copyright (C) 2022 Johannes Haug.
"""
from abc import ABCMeta
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
from float.pipeline.utils_pipeline import validate_pipeline_attrs, update_progress_bar, print_evaluation_summary
from float.prediction import BasePredictor
from float.prediction.evaluation import PredictionEvaluator
from float.prediction.river import RiverClassifier


class BasePipeline(metaclass=ABCMeta):
    """Abstract base class for evaluation pipelines.

    Attributes:
        data_loader (DataLoader): Data loader object.
        predictors (List[BasePredictor]): Predictive model(s).
        prediction_evaluators (List[PredictionEvaluator]): Evaluator(s) for the predictive model(s).
        change_detector (ConceptDriftDetector | None): Concept drift detection model.
        change_detection_evaluator (ChangeDetectionEvaluator | None): Evaluator for active concept drift detection.
        feature_selector (BaseFeatureSelector | None): Online feature selection model.
        feature_selection_evaluator (FeatureSelectionEvaluator | None): Evaluator for the online feature selection.
        batch_size (int | None): Batch size, i.e. no. of observations drawn from the data loader at one time step.
        n_pretrain (int | None): Number of observations used for the initial training of the predictive model.
        n_max (int | None): Maximum number of observations used in the evaluation.
        label_delay_range (tuple | None):
            The min and max delay in the availability of labels in time steps. The delay is sampled uniformly from
            this range.
        estimate_memory_alloc (bool):
            Boolean that indicates if the method-wise change in allocated memory (GB) shall be monitored.
            Note that this delivers only an indication of the approximate memory consumption and can significantly
            increase the total run time of the pipeline.
        test_interval (int):
            The interval/frequency at which the online learning models are evaluated. This parameter is always 1 for a
            prequential or distributed fold evaluation.
        rng (Generator): A numpy random number generator object.
        start_time (float): Physical start time.
        time_step (int): Current logical time step, i.e. iteration.
        n_total (int): Total number of observations currently observed.
    """

    def __init__(self,
                 data_loader: DataLoader,
                 predictor: Union[BasePredictor, List[BasePredictor]],
                 prediction_evaluator: PredictionEvaluator,
                 change_detector: Optional[BaseChangeDetector],
                 change_detection_evaluator: Optional[ChangeDetectionEvaluator],
                 feature_selector: Optional[BaseFeatureSelector],
                 feature_selection_evaluator: Optional[FeatureSelectionEvaluator],
                 batch_size: int,
                 n_pretrain: int,
                 n_max: int,
                 label_delay_range: Optional[tuple],
                 test_interval: int,
                 estimate_memory_alloc: bool,
                 random_state: int):
        """Initializes the pipeline.

        Args:
            data_loader: Data loader object.
            predictor: Predictor object or list of predictor objects.
            prediction_evaluator: Evaluator object for the predictive model(s).
            change_detector: Concept drift detection model.
            change_detection_evaluator: Evaluator for active concept drift detection.
            feature_selector: Online feature selection model.
            feature_selection_evaluator: Evaluator for the online feature selection.
            batch_size: Batch size, i.e. no. of observations drawn from the data loader at one time step.
            n_pretrain: Number of observations used for the initial training of the predictive model.
            n_max: Maximum number of observations used in the evaluation.
            test_interval:
                The interval/frequency at which the online learning models are evaluated. This parameter is always 1 for
                a prequential evaluation.
            estimate_memory_alloc:
                Boolean that indicates if the method-wise change in allocated memory (GB) shall be monitored.
                Note that this delivers only an indication of the approximate memory consumption and can significantly
                increase the total run time of the pipeline.
            random_state: A random integer seed used to specify a random number generator.

        Raises:
            AttributeError: If one of the provided objects is not valid.
        """
        self.data_loader = data_loader

        if isinstance(predictor, list):
            self.predictors = predictor
            self.prediction_evaluators = []  # Make copies of the PredictionEvaluator for each predictor object.
            for _ in range(len(self.predictors)):
                self.prediction_evaluators.append(copy.deepcopy(prediction_evaluator))
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
        self.test_interval = test_interval
        self.estimate_memory_alloc = estimate_memory_alloc
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

        X, y = self.data_loader.get_data(self.n_pretrain)
        self.n_total += self.n_pretrain

        for predictor in self.predictors:
            if isinstance(predictor, RiverClassifier) and not predictor.can_mini_batch:
                # Some River classifiers do not support batch-processing. In this case, we pretrain iteratively.
                for x_i, y_i in zip(X, y):
                    predictor.partial_fit(X=x_i, y=y_i)
            else:
                predictor.partial_fit(X=copy.copy(X), y=copy.copy(y))
            predictor.has_been_trained = True

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

        # ----------------------------------------
        # Concept Drift Detection
        # ----------------------------------------
        if self.change_detector is not None:
            if X_train.shape[0] > 0 and self.predictors[0].has_been_trained:
                if self.estimate_memory_alloc:
                    start_snapshot = tracemalloc.take_snapshot()

                start_time = time.time()
                if self.change_detector.error_based:
                    # Ae always use the first predictor for predicting y_pred, which is then used for change detection.
                    # Besides, we use the train set to update the change detector!
                    y_pred = self.predictors[0].predict(X_train)
                    self.change_detector.partial_fit(np.asarray(y_pred == y_train).flatten())
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
                            predictor.reset()
                    if self.change_detector is not None and self.change_detector.reset_after_drift:
                        self.change_detector.reset()

                partial_change_detected, partial_change_features = self.change_detector.detect_partial_change()
                if partial_change_detected:
                    self.change_detector.partial_drifts.append((self.time_step, partial_change_features))
            else:
                self._add_nan_measure(evaluator=self.change_detection_evaluator, is_test=False, is_train=True)

            if last_iteration:  # The concept drift detection is only evaluated in the last iteration.
                self.change_detection_evaluator.run(drifts=self.change_detector.drifts)

        # ----------------------------------------
        # Online Feature Selection
        # ----------------------------------------
        if self.feature_selector is not None:
            if X_train.shape[0] > 0:
                if self.estimate_memory_alloc:
                    start_snapshot = tracemalloc.take_snapshot()

                start_time = time.time()
                self.feature_selector.weight_features(X=copy.copy(X_train), y=copy.copy(y_train))
                self.feature_selection_evaluator.comp_times.append(time.time() - start_time)

                if self.estimate_memory_alloc:
                    self.feature_selection_evaluator.memory_changes.append(
                        self._get_memory_snapshot_diff(start_snapshot=start_snapshot))

                X_train = self.feature_selector.select_features(X=copy.copy(X_train), rng=self.rng)

                if self.time_step % self.test_interval == 0 or last_iteration:
                    self.feature_selection_evaluator.run(self.feature_selector.selected_features_history,
                                                         self.feature_selector.n_total_features)
                else:
                    self._add_nan_measure(evaluator=self.feature_selection_evaluator, is_test=True, is_train=False)
            else:
                self._add_nan_measure(evaluator=self.feature_selection_evaluator, is_test=True, is_train=True)

        # ----------------------------------------
        # Prediction
        # ----------------------------------------
        if self.predictors is not None:
            for pred_idx, (predictor, prediction_evaluator) in enumerate(zip(self.predictors, self.prediction_evaluators)):
                # Test in the specified frequency (the interval is 1 for all but the HoldoutPipeline)
                if pred_idx in predictors_for_testing \
                        and (self.time_step % self.test_interval == 0 or last_iteration) \
                        and predictor.has_been_trained:
                    start_time = time.time()
                    y_pred = predictor.predict(X_test)
                    prediction_evaluator.testing_comp_times.append(time.time() - start_time)
                    prediction_evaluator.run(y_true=copy.copy(y_test),
                                             y_pred=copy.copy(y_pred),
                                             X=copy.copy(X_test),
                                             predictor=predictor,
                                             rng=self.rng)
                else:
                    self._add_nan_measure(evaluator=prediction_evaluator, is_test=True, is_train=False)

                # Train the predictor.
                if pred_idx in predictors_for_training and X_train.shape[0] > 0:
                    if self.estimate_memory_alloc:
                        start_snapshot = tracemalloc.take_snapshot()

                    X_train_weighted = copy.copy(X_train)
                    y_train_weighted = copy.copy(y_train)
                    if predictors_training_weights is not None:
                        # Repeat/Weight training observations acc. to the specified weight.
                        X_train_weighted = np.repeat(X_train, predictors_training_weights[pred_idx], axis=0)
                        y_train_weighted = np.repeat(y_train, predictors_training_weights[pred_idx])

                    start_time = time.time()
                    predictor.partial_fit(X_train_weighted, y_train_weighted)
                    prediction_evaluator.training_comp_times.append(time.time() - start_time)

                    if not predictor.has_been_trained:
                        predictor.has_been_trained = True

                    if self.estimate_memory_alloc:
                        prediction_evaluator.memory_changes.append(
                            self._get_memory_snapshot_diff(start_snapshot=start_snapshot))
                else:
                    self._add_nan_measure(evaluator=prediction_evaluator, is_test=False, is_train=True)

    def _add_nan_measure(self,
                         evaluator: Union[PredictionEvaluator, FeatureSelectionEvaluator, ChangeDetectionEvaluator],
                         is_test: bool,
                         is_train: bool):
        """Add nan to all measures of the evaluator.

        For some pipelines and configurations (e.g. when using label delay) it can happen that a classifier is not
        trained or tested at a given iteration. In this case, we cannot compute performance measures. Instead, we append
        nan to obtain equally sized vectors at the end of training.

        Args:
            evaluator: Evaluator object.
            is_test: Indicates whether we add nan for the testing measures.
            is_test: Indicates whether we add nan for the training measures.
        """
        if is_test:
            if isinstance(evaluator, PredictionEvaluator):
                evaluator.testing_comp_times.append(np.nan)

            for measure_name in evaluator.result:
                for stat in evaluator.result[measure_name]:
                    if stat in ['mean_decay', 'var_decay']:
                        # For decayed measures we only append nan in the first time step. Otherwise, we repeat the
                        # previous measure.
                        if len(evaluator.result[measure_name][stat]) > 0:
                            evaluator.result[measure_name][stat].append(evaluator.result[measure_name][stat][-1])
                        else:
                            evaluator.result[measure_name][stat] = [np.nan]
                    else:
                        evaluator.result[measure_name][stat].append(np.nan)

        if is_train:
            if isinstance(evaluator, PredictionEvaluator):
                evaluator.training_comp_times.append(np.nan)
            else:
                evaluator.comp_times.append(np.nan)

            if self.estimate_memory_alloc:
                evaluator.memory_changes.append(np.nan)

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

    def _draw_observations(self, n_batch: int) -> Tuple[Tuple[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]]:
        """Returns the training and test set to be used for the current iteration.

        If there is no label delay, the returned train and test set are equivalent.

        Args:
            n_batch: The batch size.

        Returns:
            Tuple[Tuple[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]]:
                The training and test observations and their corresponding labels.
        """
        X, y = self.data_loader.get_data(n_batch=n_batch)
        test_set = (X, y)

        if self.label_delay_range:
            # Save observations to buffer.
            self.sample_buffer.extend(list(zip(X, y, self.time_step + self.rng.integers(self.label_delay_range[0],
                                                                                        self.label_delay_range[1],
                                                                                        X.shape[0]))))

            # Draw all available observations at current time step from buffer.
            train_set = (np.array([X for (X, _, time_step) in self.sample_buffer if time_step <= self.time_step]),
                         np.array([y for (_, y, time_step) in self.sample_buffer if time_step <= self.time_step]))
            self.sample_buffer = [(X, y, time_step) for (X, y, time_step) in self.sample_buffer
                                  if time_step > self.time_step]
        else:
            train_set = copy.deepcopy(test_set)

        return train_set, test_set

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
