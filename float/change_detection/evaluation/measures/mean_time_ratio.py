from float.change_detection.evaluation.measures.detected_change_rate import detected_change_rate
from float.change_detection.evaluation.measures.time_between_false_alarms import time_between_false_alarms
from float.change_detection.evaluation.measures.time_to_detection import time_to_detection


def mean_time_ratio(evaluator, global_drifts, n_delay):
    """
    Get the Mean Time Ratio of known concept drifts, acc. to
    Bifet, Albert, et al. "CD-MOA: Change detection framework for massive online analysis."
    International Symposium on Intelligent Data Analysis. Springer, Berlin, Heidelberg, 2013.

    mean_time_ratio = (time_between_false_alarms / time_to_detection) * detected_change_rate

    Args:
        evaluator (ChangeDetectionEvaluator): evaluator object
        global_drifts (list): time steps where a global concept drift was detected
        n_delay (int): no. of observations after a known concept drift in which to count detections as true positive
    Returns:
        float: mean time ratio
    """
    return (time_between_false_alarms(evaluator, global_drifts, n_delay)
            / time_to_detection(evaluator, global_drifts, n_delay)) * detected_change_rate(evaluator, global_drifts, n_delay)
