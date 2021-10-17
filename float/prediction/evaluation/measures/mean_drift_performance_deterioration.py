"""Drift Performance Deterioration Measure.

This function returns the drift performance deterioration. This measure corresponds to the mean difference of some
performance measure before and after a known concept drift. It is hence a measure to quantify the adaptability of a
predictor under concept drift.

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
from sklearn.metrics import zero_one_loss
from typing import Callable, Union, List
import warnings


def mean_drift_performance_deterioration(result: dict, known_drifts: Union[List[int], List[tuple]], batch_size: int,
                                         reference_measure: Callable = zero_one_loss, interval: int = 10) -> float:
    """Calculates the mean drift performance deterioration measure.

    Args:
        result: A result dictionary from the PredictionEvaluator object.
        known_drifts:
                The positions in the dataset (indices) corresponding to known concept drifts.
        batch_size: The number of observations processed per iteration/time step.
        reference_measure: Evaluation measure function
        interval:
            Scalar specifying the size of the interval (i.e. number of time steps) after known concept drift, in which
            we investigate the a performance decay of the reference measure.

    Returns:
        float: Current mean performance decay after (known) concept drifts regarding the reference measure.
    """
    init_interval = interval
    len_result = len(result[reference_measure.__name__]['measures'])

    # Get previous mean deterioration (decay)
    if len_result > 1:
        decay = result['mean_drift_performance_deterioration']['measures'][-1]
    else:
        decay = 0

    # Compute performance decay for each known drift that has already happened
    for i, kd in enumerate(known_drifts):
        if isinstance(kd, tuple):
            drift_t = round(kd[0] / batch_size)  # consider beginning of drift as reference point
        else:
            drift_t = round(kd / batch_size)

        # Check if we are still inside of the interval
        if len_result > drift_t + interval:
            continue
        elif len_result > drift_t:
            # Reset interval if we have not seen enough observations yet
            if len_result < drift_t + interval:
                interval = len_result - drift_t
            if drift_t - interval < 0:
                interval = drift_t

            if reference_measure.__name__ in result:
                before_scores = result[reference_measure.__name__]['measures'][drift_t - interval:drift_t]
                after_scores = result[reference_measure.__name__]['measures'][drift_t:drift_t + interval]
                diff_new = np.mean(after_scores) - np.mean(before_scores)
                decay = decay + (diff_new - decay) / (i * init_interval + interval)  # incremental mean
            else:
                warnings.warn('The reference measure {} is not part of the PredictionEvaluator; we return 0 per '
                              'default. Please provide {} to the PredictionEvaluator and rerun.'.format(
                    reference_measure.__name__, reference_measure.__name__))
        else:
            break  # other known drifts have not been reached yet

    return decay
