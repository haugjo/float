"""Missed Detection Rate Measure.

This function returns the rate of known concept drifts that the model has failed to detect.
The measure was introduced by
Bifet, Albert, et al. "CD-MOA: Change detection framework for massive online analysis."
International Symposium on Intelligent Data Analysis. Springer, Berlin, Heidelberg, 2013.

The missed detection rate is equal to 1 - detected_change_rate.

Copyright (C) 2022 Johannes Haug.
"""
from float.change_detection.evaluation.measures.detected_change_rate import detected_change_rate
from float.change_detection.evaluation.change_detection_evaluator import ChangeDetectionEvaluator


def missed_detection_rate(evaluator: ChangeDetectionEvaluator, drifts: list, n_delay: int) -> float:
    """Calculates the rate of missed known concept drifts.

    The missed detection rate is equal to 1 - the detected change rate.

    Args:
        evaluator: The ChangeDetectionEvaluator object.
        drifts: List of time steps corresponding to detected concept drifts.
        n_delay:
            The number of observations after a known concept drift, during which we count the detections made by the
            model as true positives.

    Returns:
        float: The fraction of missed known concept drifts.
    """
    return 1 - detected_change_rate(evaluator=evaluator, drifts=drifts, n_delay=n_delay)
