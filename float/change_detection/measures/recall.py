import numpy as np


def recall(evaluator, global_drifts, n_delay):
    """
    Get the recall of known concept drifts

    Args:
        evaluator (ChangeDetectionEvaluator): evaluator object
        global_drifts (list): time steps where a global concept drift was detected
        n_delay (int): no. of observations after a known concept drift in which to count detections as true positive
    Returns:
        float: recall of known concept drifts
    """
    if len(global_drifts) == 0:  # if there is no detected drift, the recall is zero
        return 0, 0

    iter_drifts = iter(evaluator.known_drifts)
    detections = np.asarray(global_drifts) * evaluator.batch_size  # translate drifts to relative position in dataset
    recall = 0
    drift = next(iter_drifts, None)
    while drift is not None:
        # Find start of known drift
        if isinstance(drift, tuple):  # incremental/gradual drifts involve a starting and end point
            start_search = drift[0]
        else:
            start_search = drift

        # Find end of considered search space
        drift = next(iter_drifts, None)  # get next drift
        if drift is not None:
            if isinstance(drift, tuple):  # end of search space = start of next known drift OR delay (min value)
                end_search = min(drift[0], start_search + n_delay)
            else:
                end_search = min(drift, start_search + n_delay)
        else:
            end_search = start_search + n_delay  # if no more concept drift, set end search space to max delay

        # Find first relevant detection
        relevant_drift = next((det for det in detections if start_search <= det < end_search), None)

        if relevant_drift is not None:
            recall += (1 / len(evaluator.known_drifts))

    return recall
