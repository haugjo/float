from float.change_detection.measures.detected_change_rate import detected_change_rate


def missed_detection_rate(evaluator, global_drifts, n_delay):
    """
    Get the Missed Detection Rate of known concept drifts, acc. to
    Bifet, Albert, et al. "CD-MOA: Change detection framework for massive online analysis."
    International Symposium on Intelligent Data Analysis. Springer, Berlin, Heidelberg, 2013.

    NOTE: this is equivalent to 1 - (detected_change_rate)

    Args:
        evaluator (ChangeDetectionEvaluator): evaluator object
        global_drifts (list): time steps where a global concept drift was detected
        n_delay (int): no. of observations after a known concept drift in which to count detections as true positive
    Returns:
        float: rate of missed known concept drifts
    """
    return 1 - detected_change_rate(evaluator, global_drifts, n_delay)
