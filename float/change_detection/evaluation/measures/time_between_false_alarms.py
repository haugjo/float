"""Time Between False Alarms Measure.

This function returns the mean time between false alarms as introduced in:
Bifet, Albert, et al. "CD-MOA: Change detection framework for massive online analysis."
International Symposium on Intelligent Data Analysis. Springer, Berlin, Heidelberg, 2013.

Copyright (C) 2022 Johannes Haug.
"""
import numpy as np
import warnings

from float.change_detection.evaluation.change_detection_evaluator import ChangeDetectionEvaluator


def time_between_false_alarms(evaluator: ChangeDetectionEvaluator, drifts: list, n_delay: int) -> float:
    """Calculates the mean time between false alarms.

    Args:
        evaluator: The ChangeDetectionEvaluator object.
        drifts: List of time steps corresponding to detected concept drifts.
        n_delay:
            The number of observations after a known concept drift, during which we count the detections made by the
            model as true positives.

    Returns:
        float: The mean time between false alarms in number of observations.
    """
    if len(drifts) == 0:  # If there is no detected drift, the time between false alarms is maximal.
        warnings.warn('There was no concept drift detected. In this case, the mean_time_between_false_alarms '
                      'equals the total number of observations per default.')
        return evaluator.n_total

    iter_drifts = iter(evaluator.known_drifts)
    detections = np.asarray(drifts) * evaluator.batch_size  # Translate drifts to relative position in dataset
    time_between = 0
    n_diffs = 0
    last_false_alarm = 0
    start_search = evaluator.n_init_tolerance
    drift = next(iter_drifts, None)
    while drift is not None:
        # Find end of considered search space
        if isinstance(drift, tuple):  # Incremental/gradual drifts involve a starting and end point
            end_search = drift[0]
            end_drift = drift[1]
        else:
            end_search = drift
            end_drift = drift

        relevant_drifts = [det for det in detections if start_search <= det < end_search]
        diffs = np.diff(relevant_drifts)

        # Append difference of first false positive in current range and last false positive in last range
        if len(relevant_drifts) > 0:
            if last_false_alarm != 0:
                diffs = np.append(diffs, relevant_drifts[0] - last_false_alarm)
            last_false_alarm = relevant_drifts[-1]

        time_between += np.sum(diffs)
        n_diffs += len(diffs)

        start_search = end_drift + n_delay
        drift = next(iter_drifts, None)

    # Finally, add all false discoveries after the last known drift
    relevant_drifts = [det for det in detections if det >= start_search]
    diffs = np.diff(relevant_drifts)

    if len(relevant_drifts) > 0:
        if last_false_alarm != 0:
            diffs = np.append(diffs, relevant_drifts[0] - last_false_alarm)
    time_between += np.sum(diffs)
    n_diffs += len(diffs)

    if n_diffs > 0:
        return time_between / n_diffs
    else:
        warnings.warn('There were less than two false concept drift detections. In this case, the '
                      'mean_time_between_false_alarms equals the total number of observations per default.')
        return evaluator.n_total
