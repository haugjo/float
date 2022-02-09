"""False Discovery Rate Measure.

This function returns the false discovery rate, i.e. the fraction of false positives among all detected drifts.

Copyright (C) 2022 Johannes Haug.
"""
import numpy as np

from float.change_detection.evaluation.change_detection_evaluator import ChangeDetectionEvaluator


def false_discovery_rate(evaluator: ChangeDetectionEvaluator, drifts: list, n_delay: int) -> float:
    """Calculates the false discovery rate of detected drifts.

    Args:
        evaluator: The ChangeDetectionEvaluator object.
        drifts: List of time steps corresponding to detected concept drifts.
        n_delay:
            The number of observations after a known concept drift, during which we count the detections made by the
            model as true positives.

    Returns:
        float: The false discovery rate of detected concept drifts.
    """
    if len(drifts) == 0:  # If there is no detected drift, the FDR is zero
        return 0

    iter_drifts = iter(evaluator.known_drifts)
    detections = np.asarray(drifts) * evaluator.batch_size  # Translate drifts to relative position in dataset
    false_discoveries = 0
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
        false_discoveries += len(relevant_drifts)

        # Update starting point
        start_search = end_drift + n_delay

        drift = next(iter_drifts, None)

    # Finally, add all false discoveries after the last known drift
    relevant_drifts = [det for det in detections if det >= start_search]
    false_discoveries += len(relevant_drifts)

    return false_discoveries / len(detections)
