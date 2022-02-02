"""Prequential Pipeline Module.

This module implements a pipeline following the prequential (i.e. test-then-train) evaluation strategy.

Copyright (C) 2022 Johannes Haug

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
                 estimate_memory_alloc: bool = False,
                 random_state: int = 0):
        """Initializes the pipeline.

        Args:
            data_loader: Data loader object.
            predictor: Predictive model(s).
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
                         test_interval=1,  # Defaults to one for a prequential evaluation.
                         estimate_memory_alloc=estimate_memory_alloc,
                         random_state=random_state)

    def run(self):
        """Runs the pipeline."""
        super().run()

        # Run the prequential evaluation.
        last_iteration = False

        while self.n_total < self.n_max:
            n_batch = self._get_n_batch()

            if self.n_total + n_batch >= self.n_max:
                last_iteration = True

            train_set, test_set = self._draw_observations(n_batch=n_batch)

            try:
                self._run_iteration(train_set=train_set,
                                    test_set=test_set,
                                    last_iteration=last_iteration,
                                    predictors_for_testing=list(np.arange(len(self.predictors))),  # Use all predictors.
                                    predictors_for_training=list(np.arange(len(self.predictors))))
            except BaseException:  # This exception is left unspecific on purpose to fetch all possible errors.
                traceback.print_exc()
                break

            self._finish_iteration(n_batch=n_batch)

        self._finish_evaluation()
