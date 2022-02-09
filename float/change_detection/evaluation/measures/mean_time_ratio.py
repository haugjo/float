"""Mean Time Ratio Measure.

This function returns the mean time ration measure as introduced in:
Bifet, Albert, et al. "CD-MOA: Change detection framework for massive online analysis."
International Symposium on Intelligent Data Analysis. Springer, Berlin, Heidelberg, 2013.

Copyright (C) 2022 Johannes Haug.
"""
from float.change_detection.evaluation.measures.detected_change_rate import detected_change_rate
from float.change_detection.evaluation.measures.time_between_false_alarms import time_between_false_alarms
from float.change_detection.evaluation.measures.detection_delay import detection_delay
from float.change_detection.evaluation.change_detection_evaluator import ChangeDetectionEvaluator


def mean_time_ratio(evaluator: ChangeDetectionEvaluator, drifts: list, n_delay: int) -> float:
    """Calculates the mean time ratio.

    The mean time ratio is a function of the detection delay, the time between false alarms,
    and the detected change rate:
    mean_time_ratio = (time_between_false_alarms / detection_delay) * detected_change_rate

    Args:
        evaluator: The ChangeDetectionEvaluator object.
        drifts: List of time steps corresponding to detected concept drifts.
        n_delay:
            The number of observations after a known concept drift, during which we count the detections made by the
            model as true positives.

    Returns:
        float: The mean time ratio.
    """
    return (time_between_false_alarms(evaluator=evaluator, drifts=drifts, n_delay=n_delay)
            / detection_delay(evaluator=evaluator, drifts=drifts, n_delay=n_delay)) \
           * detected_change_rate(evaluator=evaluator, drifts=drifts, n_delay=n_delay)
