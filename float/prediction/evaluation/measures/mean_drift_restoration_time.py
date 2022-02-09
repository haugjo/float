"""Drift Restoration Time Measure.

This function returns the mean drift restoration time, i.e. the average number of iterations (time steps) after a
known concept drift, before the previous performance has been restored. It is hence a measure to quantify the
adaptability of a predictor under concept drift.

Copyright (C) 2022 Johannes Haug.
"""
import numpy as np
import warnings
from sklearn.metrics import zero_one_loss
from typing import Union, List, Callable, Optional


def mean_drift_restoration_time(result: dict,
                                known_drifts: Union[List[int], List[tuple]],
                                batch_size: int,
                                reference_measure: Callable = zero_one_loss,
                                reference_measure_kwargs: Optional[dict] = None,
                                incr: bool = False,
                                interval: int = 10) -> float:
    """Calculates the mean restoration time after known concept drifts.

    Args:
        result: A result dictionary from the PredictionEvaluator object.
        known_drifts:
                The positions in the dataset (indices) corresponding to known concept drifts.
        batch_size: The number of observations processed per iteration/time step.
        reference_measure: Evaluation measure function.
        reference_measure_kwargs:
            Keyword arguments of the reference measure. This attribute is maintained for consistency reasons, but is
            not used by this performance measure.
        incr: Boolean indicating whether the evaluation measure is incremental (i.e. higher is better).
        interval:
            Scalar specifying the size of the interval (i.e. number of time steps) after known concept drift, in which
            we investigate a performance decay of the reference measure.

    Returns:
        float: Current mean no. of iterations before recovery from (known) concept drifts.
    """
    iter_drifts = reversed(known_drifts)
    len_result = len(result[reference_measure.__name__]['measures'])

    # Get previous mean recovery time
    if len_result > 1:
        recovery = result['mean_drift_restoration_time']['measures'][-1]
    else:
        return 0

    # Identify currently relevant drift
    drift = next(iter_drifts, None)
    i = len(known_drifts)  # Drift identifier
    while drift is not None:
        if isinstance(drift, tuple):
            drift = round(drift[0] / batch_size)  # Consider beginning of drift as reference point
        else:
            drift = round(drift / batch_size)

        if len_result > drift:
            break
        else:
            i -= 1
            drift = next(iter_drifts, None)

    if drift is not None:
        if len_result > drift + 1 and recovery == result['mean_drift_restoration_time']['measures'][-2]:
            return recovery  # Return, if model has already recovered from drift
        else:
            # Get current recovery time
            if reference_measure.__name__ in result:
                value_before = np.mean(result[reference_measure.__name__]['measures'][drift - interval:drift])
                if incr:
                    relevant_idx = np.argwhere(result[reference_measure.__name__]['measures'][drift:] >= value_before).flatten()
                else:
                    relevant_idx = np.argwhere(result[reference_measure.__name__]['measures'][drift:] <= value_before).flatten()

                if len(relevant_idx) > 0:
                    rec_new = relevant_idx[0]
                else:  # Model has not recovered yet, use time since drift as return value
                    rec_new = len(result[reference_measure.__name__]['measures'][drift:])

                # Get recovery time prior to current drift
                prev_drift_recovery = result['mean_drift_restoration_time']['measures'][drift - 1]
                recovery = prev_drift_recovery + (rec_new - prev_drift_recovery) / i  # Incremental mean
            else:
                warnings.warn("The reference measure {} is not part of the PredictionEvaluator; we return 0 per "
                              "default. Please provide {} to the PredictionEvaluator and rerun.".format(
                    reference_measure.__name__, reference_measure.__name__))

    return recovery
