import numpy as np


def detection_delay(evaluator, global_drifts, n_delay):
    """
    Get the mean Time to Detection,
    i.e. the average delay between the beginning of a known drift and the first detection.

    This measure is introduced in
    Bifet, Albert, et al. "CD-MOA: Change detection framework for massive online analysis."
    International Symposium on Intelligent Data Analysis. Springer, Berlin, Heidelberg, 2013.

    Args:
        evaluator (ChangeDetectionEvaluator): evaluator object
        global_drifts (list): time steps where a global concept drift was detected
        n_delay (int): no. of observations after a known concept drift in which to count detections as true positive (not required for this measure!)
    Returns:
        float: avg. delay in no. of observations between a known drift and the first detected drift
    """
    iter_drifts = iter(evaluator.known_drifts)
    detections = np.asarray(global_drifts) * evaluator.batch_size  # translate drifts to relative position in dataset
    delay = 0
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
            if isinstance(drift, tuple):  # end of search space = start of next known drift
                end_search = drift[0]
            else:
                end_search = drift
        else:
            end_search = evaluator.n_samples  # if no more concept drift, set end search space to n_samples

        # Find first relevant drift detection
        relevant_drift = next((det for det in detections if start_search <= det < end_search), None)

        if relevant_drift is not None:
            delay += relevant_drift - start_search  # add delay
        else:
            delay += end_search - start_search  # add default value

    return delay / len(evaluator.known_drifts)
