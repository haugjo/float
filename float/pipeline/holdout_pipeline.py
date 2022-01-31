"""Periodic Holdout Pipeline Module.

This module implements a pipeline following the periodic holdout evaluation strategy.

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
from numpy.typing import ArrayLike
import traceback
import warnings
from typing import Optional, Union, List, Tuple

from float.pipeline.base_pipeline import BasePipeline
from float.data.data_loader import DataLoader
from float.feature_selection import BaseFeatureSelector
from float.feature_selection.evaluation import FeatureSelectionEvaluator
from float.change_detection import BaseChangeDetector
from float.change_detection.evaluation import ChangeDetectionEvaluator
from float.prediction import BasePredictor
from float.prediction.evaluation import PredictionEvaluator


class HoldoutPipeline(BasePipeline):
    """Pipeline class for periodic holdout evaluation.

    Attributes:
        test_set (Tuple[ArrayLike, ArrayLike] | None):
            A tuple containing the initial test observations and labels used for the holdout evaluation.
        test_replace_interval (int | None):
                This integer specifies in which interval we replace the oldest test observation. For example, if
                test_replace_interval=10 then we will use every 10th observation to replace the currently oldest test
                observation. Note that test observations will not be used for training, hence this interval should not
                be chosen too small. If argument is None, we use the complete batch at testing time in the evluation.
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
                 test_set: Optional[Tuple[ArrayLike, ArrayLike]] = None,
                 test_interval: int = 10,
                 test_replace_interval: Optional[int] = None,
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
            test_set: A tuple containing the initial test observations and labels used for the holdout evaluation.
            test_interval: The interval/frequency at which the online learning models are evaluated.
            test_replace_interval:
                This integer specifies in which interval we replace the oldest test observation. For example, if
                test_replace_interval=10 then we will use every 10th observation to replace the currently oldest test
                observation. Note that test observations will not be used for training, hence this interval should not
                be chosen too small. If argument is None, we use the complete batch at testing time in the evaluation.
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
                         test_interval=test_interval,
                         estimate_memory_alloc=estimate_memory_alloc,
                         random_state=random_state)

        self.test_set = test_set
        self.test_replace_interval = test_replace_interval
        if self.test_set is None:
            warnings.warn("No initial test set has been provided. By default, the holdout pipeline will use a test set "
                          "with the same size as the batch size. If the replace interval is None, the pipeline will "
                          "use the current data batch for testing, resembling a prequential evaluation.")
            self._test_set_size = self.batch_size
        else:
            self._test_set_size = self.test_set[0].shape[0]

        if self.test_replace_interval is None:
            warnings.warn("The test_replace_interval is None, which means that the test set will not be updated "
                          "over time. This can lead to invalid results!")

    def run(self):
        """ Runs the pipeline."""
        super().run()

        # Run the holdout evaluation.
        last_iteration = False

        while self.n_total < self.n_max:
            n_batch = self._get_n_batch()

            if self.n_total + n_batch >= self.n_max:
                last_iteration = True

            (X_train, y_train), (X_test, y_test) = self._draw_observations(n_batch=n_batch)

            if self.test_replace_interval is None:
                if self.test_set is not None:
                    X_test, y_test = self.test_set
            else:
                # Check if we need to replace the oldest test instance in this iteration.
                mods = (np.arange(1, X_train.shape[0] + 1) + self.n_total) % self.test_replace_interval
                new_test_X = X_train[mods == 0]
                new_test_y = y_train[mods == 0]

                X_train = X_train[mods != 0]  # Drop test instances from the training set.
                y_train = y_train[mods != 0]

                if self.test_set is not None:
                    X_test, y_test = self.test_set
                    X_test = np.append(X_test, new_test_X, axis=0)
                    y_test = np.append(y_test, new_test_y, axis=0)
                else:
                    X_test = new_test_X
                    y_test = new_test_y

                if X_test.shape[0] >= self._test_set_size:  # Drop old instances.
                    n_remove = X_test.shape[0] - self._test_set_size
                    X_test = X_test[n_remove:, :]
                    y_test = y_test[n_remove:]

            train_set = (X_train, y_train)
            self.test_set = (X_test, y_test)

            try:
                self._run_iteration(train_set=train_set,
                                    test_set=self.test_set,
                                    last_iteration=last_iteration,
                                    predictors_for_testing=list(np.arange(len(self.predictors))),  # Use all predictors.
                                    predictors_for_training=list(np.arange(len(self.predictors))))
            except BaseException:  # This exception is left unspecific on purpose to fetch all possible errors.
                traceback.print_exc()
                break

            self._finish_iteration(n_batch=n_batch)

        self._finish_evaluation()
