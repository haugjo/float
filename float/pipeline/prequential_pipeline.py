"""Prequential Pipeline Module.

This module implements a pipeline following the prequential (i.e. test-then-train) evaluation strategy.

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


class PrequentialPipeline(BasePipeline):
    """Pipeline class for prequential evaluation."""
    def __init__(self, data_loader: DataLoader,
                 predictor: Optional[BasePredictor] = None,
                 prediction_evaluator: Optional[PredictionEvaluator] = None,
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
                 random_state: int = 0):
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
            label_delay_range:
                The min and max delay in the availability of labels in time steps. The delay is sampled uniformly from
                this range.
            known_drifts: The positions in the dataset (indices) corresponding to known concept drifts.
            estimate_memory_alloc:
                Boolean that indicates if the method-wise change in allocated memory (GB) shall be monitored.
                Note that this delivers only an indication of the approximate memory consumption and can significantly
                increase the total run time of the pipeline.
            random_state: A random integer seed used to specify a random number generator.
        """
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
                         known_drifts=known_drifts,
                         estimate_memory_alloc=estimate_memory_alloc,
                         test_interval=1,  # Defaults to one for a prequential evaluation.
                         random_state=random_state)

    def run(self):
        """Runs the pipeline."""
        if (self.data_loader.stream.n_remaining_samples() > 0) and \
                (self.data_loader.stream.n_remaining_samples() < self.n_max):
            self.n_max = self.data_loader.stream.n_samples
            warnings.warn("Parameter n_max exceeds the size of data_loader and will be automatically reset.",
                          stacklevel=2)

        self._start_evaluation()
        self._run_prequential()
        self._finish_evaluation()

    def _run_prequential(self):
        """Runs the prequential evaluation strategy.

        Raises:
            BaseException: If the prequential evaluation runs into an error.
        """
        while self.n_total < self.n_max:
            last_iteration = False
            n_batch = self._get_n_batch()

            if self.n_total + n_batch >= self.n_max:
                last_iteration = True

            train_set = self._get_train_set(n_batch)

            try:
                self._run_iteration(train_set=train_set, last_iteration=last_iteration)
            except BaseException:  # This exception is left unspecific on purpose to fetch all possible errors.
                traceback.print_exc()
                break

            self._finish_iteration(n_batch=n_batch)
