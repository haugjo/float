"""Detected Change Rate Measure.

This function returns the fraction of correctly detected known drifts. The detected change rate measure is sometimes
also called recall or false positive rate.

Copyright (C) 2022 Johannes Haug.
"""
import numpy as np

from float.change_detection.evaluation.change_detection_evaluator import ChangeDetectionEvaluator


def detected_change_rate(evaluator: ChangeDetectionEvaluator, drifts: list, n_delay: int) -> float:
    """Calculates the rate of correctly detected known concept drifts.

    Args:
        evaluator: The ChangeDetectionEvaluator object.
        drifts: List of time steps corresponding to detected concept drifts.
        n_delay:
            The number of observations after a known concept drift, during which we count the detections made by the
            model as true positives.

    Returns:
        float: The rate of correctly detected known concept drifts
    """
    if len(drifts) == 0:  # If there is no detected drift, the measure is zero
        return 0

    iter_drifts = iter(evaluator.known_drifts)
    detections = np.asarray(drifts) * evaluator.batch_size  # Translate drifts to relative position in dataset
    recall = 0
    drift = next(iter_drifts, None)
    while drift is not None:
        # Find start of a known drift
        if isinstance(drift, tuple):  # Incremental/gradual drifts involve a starting and end point
            start_search = drift[0]
            drift_end = drift[1]
        else:
            start_search = drift
            drift_end = drift

        # Find end of considered search space
        drift = next(iter_drifts, None)
        if drift is not None:
            if isinstance(drift, tuple):  # End of search space = start of next known drift OR delay (min value)
                end_search = min(drift[0], drift_end + n_delay)
            else:
                end_search = min(drift, drift_end + n_delay)
        else:
            end_search = drift_end + n_delay  # If no more known concept drifts, set end search space to max delay

        # Find first relevant detection
        relevant_drift = next((det for det in detections if start_search <= det < end_search), None)

        if relevant_drift is not None:
            recall += (1 / len(evaluator.known_drifts))

    return recall
