"""Distributed Fold Pipeline Module.

This module implements a pipeline following the k-fold distributed validation
techniques proposed by Albert Bifet, Gianmarco de Francisci Morales, Jesse Read,
Geoff Holmes, and Bernhard Pfahringer. 2015. Efficient Online Evaluation of Big
Data Stream Classifiers. In Proceedings of the 21th ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining (KDD '15). Association for
Computing Machinery, New York, NY, USA, 59–68.

The following three modes are implemented:
1. k-fold distributed cross-validation: each example is used for testing in one
classifier selected randomly, and used for training by all the others;
2. k-fold distributed split-validation: each example is used for training in one
classifier selected randomly, and for testing in the other classifiers;
3. k-fold distributed bootstrap validation: each example is used for training in
each classifier according to a weight from a Poisson(1) distribution. This results
in each example being used for training in approximately two thirds of the classifiers,
 with a separate weight in each classifier, and for testing in the rest.

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
import numpy as np
import traceback
import warnings
from typing import Optional, Union, List

from float.pipeline.base_pipeline import BasePipeline
from float.data.data_loader import DataLoader
from float.feature_selection import BaseFeatureSelector
from float.feature_selection.evaluation import FeatureSelectionEvaluator
from float.change_detection import BaseChangeDetector
from float.change_detection.evaluation import ChangeDetectionEvaluator
from float.prediction import BasePredictor
from float.prediction.evaluation import PredictionEvaluator


class DistributedFoldPipeline(BasePipeline):
    """Pipeline for k-fold distributed validation."""
    def __init__(self, data_loader: DataLoader,
                 predictors: List[BasePredictor],
                 prediction_evaluators: List[PredictionEvaluator],
                 change_detector: Optional[BaseChangeDetector] = None,
                 change_detection_evaluator: Optional[ChangeDetectionEvaluator] = None,
                 feature_selector: Optional[BaseFeatureSelector] = None,
                 feature_selection_evaluator: Optional[FeatureSelectionEvaluator] = None,
                 batch_size: int = 1,
                 n_pretrain: int = 100,
                 n_max: int = np.inf,
                 label_delay_range: Optional[tuple] = None,
                 known_drifts: Optional[Union[List[int], List[tuple]]] = None,
                 estimate_memory_alloc: bool = False,
                 validation_mode: str = 'cross',
                 random_state: int = 0):
        """Initializes the pipeline.

        Args:
            data_loader: Data loader object.
            predictors: Predictive models.
            prediction_evaluators: Evaluators for predictive models.
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
            known_drifts: The positions in the dataset (indices) corresponding to known concept drifts.
            estimate_memory_alloc:
                Boolean that indicates if the method-wise change in allocated memory (GB) shall be monitored.
                Note that this delivers only an indication of the approximate memory consumption and can significantly
                increase the total run time of the pipeline.
            validation_mode:
                A string indicating the k-fold distributed validation mode to use. One of 'cross', 'batch' and
                'bootstrap'.
            random_state: A random integer seed used to specify a random number generator.
        """
        self.validation_mode = validation_mode
        super().__init__(data_loader=data_loader,
                         predictors=predictors,
                         prediction_evaluators=prediction_evaluators,
                         change_detector=change_detector,
                         change_detection_evaluator=change_detection_evaluator,
                         feature_selector=feature_selector,
                         feature_selection_evaluator=feature_selection_evaluator,
                         batch_size=batch_size,
                         n_pretrain=n_pretrain,
                         n_max=n_max,
                         label_delay_range=label_delay_range,
                         known_drifts=known_drifts,
                         estimate_memory_alloc=estimate_memory_alloc,
                         test_interval=1,  # Defaults to one for a distributed fold evaluation.
                         random_state=random_state)

    def _validate(self):
        super()._validate()
        if self.validation_mode not in ['cross', 'split', 'bootstrap']:
            raise AttributeError('Please choose one of the validation modes "cross", "split", or "bootstrap".')

    def run(self):
        """Runs the pipeline."""
        if (self.data_loader.stream.n_remaining_samples() > 0) and \
                (self.data_loader.stream.n_remaining_samples() < self.n_max):
            self.n_max = self.data_loader.stream.n_samples
            warnings.warn("Parameter n_max exceeds the size of data_loader and will be automatically reset.",
                          stacklevel=2)

        self._start_evaluation()
        self._run_distributed_fold()
        self._finish_evaluation()

    def _run_distributed_fold(self):
        """Runs the distributed fold evaluation strategy.

        Raises:
            BaseException: If the distributed fold evaluation runs into an error.
        """
        while self.n_total < self.n_max:
            last_iteration = False
            n_batch = self._get_n_batch()

            if self.n_total + n_batch >= self.n_max:
                last_iteration = True

            train_set = self._get_train_set(n_batch)

            predictor_train_idx = []
            predictor_test_idx = []
            weights = None
            if self.validation_mode == 'cross':
                predictor_test_idx = [np.random.randint(0, len(self.predictors))]
                predictor_train_idx = [i for i in range(len(self.predictors)) if i not in predictor_test_idx]
            elif self.validation_mode == 'split':
                predictor_train_idx = [np.random.randint(0, len(self.predictors))]
                predictor_test_idx = [i for i in range(len(self.predictors)) if i not in predictor_train_idx]
            elif self.validation_mode == 'bootstrap':
                weights = np.random.poisson(1, len(self.predictors))
                predictor_test_idx = [i for i in range(len(self.predictors)) if weights[i] == 0]
                predictor_train_idx = [i for i in range(len(self.predictors)) if weights[i] != 0]
                weights = [i for i in weights if i != 0]

            try:
                self._run_iteration(predictor_train_idx=predictor_train_idx, predictor_test_idx=predictor_test_idx,
                                    train_weights=weights, train_set=train_set, last_iteration=last_iteration)
            except BaseException:  # This exception is left unspecific on purpose to fetch all possible errors.
                traceback.print_exc()
                break

            self._finish_iteration(n_batch=n_batch)
