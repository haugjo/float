"""Drift Performance Deterioration Measure.

This function returns the drift performance deterioration. This measure corresponds to the mean difference of a
performance measure before and after a known concept drift. It is hence a measure to quantify the adaptability of a
predictor under concept drift.

Copyright (C) 2022 Johannes Haug.
"""
import numpy as np
from sklearn.metrics import zero_one_loss
from typing import Callable, Union, List, Optional
import warnings


def mean_drift_performance_deterioration(result: dict,
                                         known_drifts: Union[List[int], List[tuple]],
                                         batch_size: int,
                                         reference_measure: Callable = zero_one_loss,
                                         reference_measure_kwargs: Optional[dict] = None,
                                         interval: int = 10) -> float:
    """Calculates the mean performance deterioration after kown concept drifts.

    Args:
        result: A result dictionary from the PredictionEvaluator object.
        known_drifts:
                The positions in the dataset (indices) corresponding to known concept drifts.
        batch_size: The number of observations processed per iteration/time step.
        reference_measure: Evaluation measure function.
        reference_measure_kwargs:
            Keyword arguments of the reference measure. This attribute is maintained for consistency reasons, but is
            not used by this performance measure.
        interval:
            Scalar specifying the size of the interval (i.e. number of time steps) after known concept drift, in which
            we investigate a performance decay of the reference measure.

    Returns:
        float: Current mean performance deterioration after (known) concept drifts regarding the reference measure.
    """
    init_interval = interval
    len_result = len(result[reference_measure.__name__]['measures'])

    # Get previous mean deterioration
    if len_result > 1:
        deter = result['mean_drift_performance_deterioration']['measures'][-1]
    else:
        deter = 0

    # Compute performance decay for each known drift that has already happened
    for i, kd in enumerate(known_drifts):
        if isinstance(kd, tuple):
            drift_t = round(kd[0] / batch_size)  # Consider beginning of drift as reference point
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
                deter = deter + (diff_new - deter) / (i * init_interval + interval)  # Incremental mean
            else:
                warnings.warn("The reference measure {} is not part of the PredictionEvaluator; we return 0 per "
                              "default. Please provide {} to the PredictionEvaluator and rerun.".format(
                    reference_measure.__name__, reference_measure.__name__))
        else:
            break  # Return, since other known drifts have not been reached yet

    return deter
