import numpy as np


def time_between_false_alarms(evaluator, global_drifts, n_delay):
    """
    Get the mean Time Between False Alarms, i.e. the mean no. of observations between false positive drift detections.

    This measure is introduced in
    Bifet, Albert, et al. "CD-MOA: Change detection framework for massive online analysis."
    International Symposium on Intelligent Data Analysis. Springer, Berlin, Heidelberg, 2013.

    Args:
        evaluator (ChangeDetectionEvaluator): evaluator object
        global_drifts (list): time steps where a global concept drift was detected
        n_delay (int): no. of observations after a known concept drift in which to count detections as true positive
    Returns:
        float: mean time between false alarms
    """
    if len(global_drifts) == 0:  # if there is no detected drift, the time between false alarms is zero
        return 0

    iter_drifts = iter(evaluator.known_drifts)
    detections = np.asarray(global_drifts) * evaluator.batch_size  # translate drifts to relative position in dataset
    time_between = 0
    n_diffs = 0
    last_false_alarm = 0
    start_search = evaluator.n_init_tolerance
    drift = next(iter_drifts, None)
    while drift is not None:
        # Find end of considered search space
        if isinstance(drift, tuple):  # incremental/gradual drifts involve a starting and end point
            end_search = drift[0]
        else:
            end_search = drift

        relevant_drifts = [det for det in detections if start_search <= det < end_search]
        diffs = np.diff(relevant_drifts)

        # Append difference of first false positive in current range and last false positive in last range
        if len(relevant_drifts) > 0:
            if last_false_alarm != 0:
                diffs = np.append(diffs, relevant_drifts[0] - last_false_alarm)
            last_false_alarm = relevant_drifts[-1]  # update last false alarm
        time_between += np.sum(diffs)
        n_diffs += len(diffs)

        # Update starting point
        start_search = end_search + n_delay

        drift = next(iter_drifts, None)  # get next drift

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
        return 0
