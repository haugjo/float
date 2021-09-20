import numpy as np


def false_discovery_rate(evaluator, global_drifts, n_delay):
    """
    Get the False Discovery Rate of the detected drifts, i.e. the fraction of false positive drift detections

    Args:
        evaluator (ChangeDetectionEvaluator): evaluator object
        global_drifts (list): time steps where a global concept drift was detected
        n_delay (int): no. of observations after a known concept drift in which to count detections as true positive
    Returns:
        float: false discovery rate of detected concept drifts
    """
    if len(global_drifts) == 0:  # if there is no detected drift, the FDR is zero
        return 0

    iter_drifts = iter(evaluator.known_drifts)
    detections = np.asarray(global_drifts) * evaluator.batch_size  # translate drifts to relative position in dataset
    false_discoveries = 0
    start_search = evaluator.n_init_tolerance
    drift = next(iter_drifts, None)
    while drift is not None:
        # Find end of considered search space
        if isinstance(drift, tuple):  # incremental/gradual drifts involve a starting and end point
            end_search = drift[0]
        else:
            end_search = drift

        relevant_drifts = [det for det in detections if start_search <= det < end_search]
        false_discoveries += len(relevant_drifts)

        # Update starting point
        start_search = end_search + n_delay

        drift = next(iter_drifts, None)  # get next drift

    # Finally, add all false discoveries after the last known drift
    relevant_drifts = [det for det in detections if det >= start_search]
    false_discoveries += len(relevant_drifts)

    return false_discoveries / len(detections)
