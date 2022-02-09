"""Distributed Fold Pipeline Module.

This module contains a pipeline that performs a k-fold distributed validation as proposed by
Albert Bifet, Gianmarco de Francisci Morales, Jesse Read,
Geoff Holmes, and Bernhard Pfahringer. 2015. Efficient Online Evaluation of Big
Data Stream Classifiers. In Proceedings of the 21th ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining (KDD '15). Association for
Computing Machinery, New York, NY, USA, 59–68.

The distributed fold pipeline maintains multiple parallel instances of each provided predictor object and enables
more robust and statistically significant results. The following three modes defined by Bifet et al. are implemented:
1. k-fold distributed cross-validation: each example is used for testing in one classifier instance selected randomly,
and used for training by all the others;
2. k-fold distributed split-validation: each example is used for training in one classifier instance selected randomly,
and for testing in the other classifiers;
3. k-fold distributed bootstrap validation: each example is used for training in each classifier instance according to a
weight from a Poisson(1) distribution. This results in each example being used for training in approximately two thirds
of the classifiers instances,  with a separate weight in each classifier, and for testing in the rest.

Copyright (C) 2022 Johannes Haug.
"""
import copy

import numpy as np
import traceback
from typing import Optional, Union, List
import warnings

from float.pipeline.base_pipeline import BasePipeline
from float.data.data_loader import DataLoader
from float.feature_selection import BaseFeatureSelector
from float.feature_selection.evaluation import FeatureSelectionEvaluator
from float.change_detection import BaseChangeDetector
from float.change_detection.evaluation import ChangeDetectionEvaluator
from float.prediction import BasePredictor
from float.prediction.evaluation import PredictionEvaluator


class DistributedFoldPipeline(BasePipeline):
    """Pipeline class for a k-fold distributed evaluation.

    Attributes:
        validation_mode (str):
            A string indicating the k-fold distributed validation mode to use. One of 'cross', 'batch' and 'bootstrap'.
        n_parallel_instances (int): The number of instances of the specified predictor that will be trained in parallel.
        n_unique_predictors (int): The number of predictor objects originally specified.
        predictors List[BasePredictor]: All instances of the predictive model(s).
        prediction_evaluators List[PredictionEvaluator]: Evaluator(s) for all instances of the predictive model(s).
    """
    def __init__(self, data_loader: DataLoader,
                 predictor: Union[BasePredictor, List[BasePredictor]],
                 prediction_evaluator: PredictionEvaluator,
                 change_detector: Optional[BaseChangeDetector] = None,
                 change_detection_evaluator: Optional[ChangeDetectionEvaluator] = None,
                 feature_selector: Optional[BaseFeatureSelector] = None,
                 feature_selection_evaluator: Optional[FeatureSelectionEvaluator] = None,
                 batch_size: int = 1,
                 n_pretrain: int = 100,
                 n_max: int = np.inf,
                 label_delay_range: Optional[tuple] = None,
                 validation_mode: str = 'cross',
                 n_parallel_instances: int = 2,
                 estimate_memory_alloc: bool = False,
                 random_state: int = 0):
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
            label_delay_range:
                The min and max delay in the availability of labels in time steps. The delay is sampled uniformly from
                this range.
            validation_mode:
                A string indicating the k-fold distributed validation mode to use. One of 'cross', 'split' and
                'bootstrap'.
            n_parallel_instances:
                The number of instances of the specified predictor that will be trained in parallel.
            estimate_memory_alloc:
                Boolean that indicates if the method-wise change in allocated memory (GB) shall be monitored.
                Note that this delivers only an indication of the approximate memory consumption and can significantly
                increase the total run time of the pipeline.
            random_state: A random integer seed used to specify a random number generator.
        """
        self.validation_mode = validation_mode
        self.n_parallel_instances = n_parallel_instances
        self._validate()

        super().__init__(data_loader=data_loader,
                         predictor=predictor,
                         prediction_evaluator=prediction_evaluator,
                         change_detector=change_detector,
                         change_detection_evaluator=change_detection_evaluator,
                         feature_selector=feature_selector,
                         feature_selection_evaluator=feature_selection_evaluator,
                         batch_size=batch_size,
                         n_pretrain=n_pretrain,
                         n_max=n_max,
                         label_delay_range=label_delay_range,
                         test_interval=1,  # Defaults to one for a distributed fold evaluation.
                         estimate_memory_alloc=estimate_memory_alloc,
                         random_state=random_state)

        # Create multiple instances of the predictor(s) and evaluator(s)
        self.n_unique_predictors = len(self.predictors)
        dist_val_predictors = []
        dist_val_evaluators = []
        for predictor, prediction_evaluator in zip(self.predictors, self.prediction_evaluators):
            dist_val_predictors.extend([copy.deepcopy(predictor) for _ in range(self.n_parallel_instances)])
            dist_val_evaluators.extend([copy.deepcopy(prediction_evaluator) for _ in range(self.n_parallel_instances)])
        self.predictors = dist_val_predictors
        self.prediction_evaluators = dist_val_evaluators

    def run(self):
        """Runs the pipeline."""
        super().run()

        # Run the distributed fold evaluation.
        last_iteration = False

        while self.n_total < self.n_max:
            n_batch = self._get_n_batch()

            if self.n_total + n_batch >= self.n_max:
                last_iteration = True

            train_set, test_set = self._draw_observations(n_batch=n_batch)

            predictors_for_testing = []
            predictors_for_training = []
            predictors_training_weights = None
            if self.validation_mode == 'cross':
                # "Each example is used for testing in one classifier selected randomly, and used for training by all
                # the others." (Bifet et al. 2015)
                for p_idx in range(self.n_unique_predictors):
                    p_test_idx = np.asarray([self.rng.integers(self.n_parallel_instances)])
                    p_train_idx = np.setdiff1d(np.arange(self.n_parallel_instances), p_test_idx)
                    predictors_for_testing.extend(p_test_idx + p_idx * self.n_parallel_instances)
                    predictors_for_training.extend(p_train_idx + p_idx * self.n_parallel_instances)
            elif self.validation_mode == 'split':
                # "Each example is used for training in one classifier selected randomly, and for testing in the
                # other classifiers." (Bifet et al. 2015)
                for p_idx in range(self.n_unique_predictors):
                    p_train_idx = np.asarray([self.rng.integers(self.n_parallel_instances)])
                    p_test_idx = np.setdiff1d(np.arange(self.n_parallel_instances), p_train_idx)
                    predictors_for_testing.extend(p_test_idx + p_idx * self.n_parallel_instances)
                    predictors_for_training.extend(p_train_idx + p_idx * self.n_parallel_instances)
            elif self.validation_mode == 'bootstrap':
                # "Each example is used for training in each classifier according to a weight from a Poisson(1)
                # distribution. This results in each example being used for training in approximately two thirds of
                # the classifiers, with a separate weight in each classifier, and for testing in the rest."
                # (Bifet et al. 2015)
                predictors_training_weights = []
                for p_idx in range(self.n_unique_predictors):
                    weights = self.rng.poisson(1, self.n_parallel_instances)
                    predictors_for_testing.extend(np.argwhere(weights == 0).flatten()
                                                  + p_idx * self.n_parallel_instances)
                    predictors_for_training.extend(np.argwhere(weights != 0).flatten()
                                                   + p_idx * self.n_parallel_instances)
                    predictors_training_weights.extend(weights)

            try:
                self._run_iteration(train_set=train_set,
                                    test_set=test_set,
                                    last_iteration=last_iteration,
                                    predictors_for_testing=predictors_for_testing,
                                    predictors_for_training=predictors_for_training,
                                    predictors_training_weights=predictors_training_weights)
            except BaseException:  # This exception is left unspecific on purpose to fetch all possible errors.
                traceback.print_exc()
                break

            self._finish_iteration(n_batch=n_batch)

        self._finish_evaluation()

    def _finish_evaluation(self):
        """Finishes the Distributed Fold Pipeline Evaluation.

        For ease of implementation, we have cloned the initially provided predictors to train multiple instances in
        parallel. Accordingly, the predictors and predictor_evaluators are one long list of classifier instances.
        For the final result, we regroup the instances per unique classifier, returning a more intuitive list of lists.
        """
        final_predictors = []
        final_prediction_evaluators = []
        for p_idx in range(self.n_unique_predictors):
            final_predictors.append(self.predictors[p_idx * self.n_parallel_instances:
                                                    (p_idx + 1) * self.n_parallel_instances])
            final_prediction_evaluators.append(self.prediction_evaluators[p_idx * self.n_parallel_instances:
                                                                          (p_idx + 1) * self.n_parallel_instances])

        self.predictors = final_predictors
        self.prediction_evaluators = final_prediction_evaluators

        super()._finish_evaluation()

    def _validate(self):
        """Validates the additional attributes of the pipeline.

        Raises:
            AttributeError: If the specified validation_mode is not implemented.
        """
        if self.validation_mode not in ['cross', 'split', 'bootstrap']:
            raise AttributeError(
                'Please choose one of the validation modes "cross", "split", or "bootstrap" that are '
                'provided by the DistributedFoldPipeline.')

        if self.n_parallel_instances < 2:
            warnings.warn('The DistributedFoldPipeline should use at least two instances of each provided '
                          'predictor for valid results. If you want to run a single instance, consider using the '
                          'PrequentialPipeline instead.')
