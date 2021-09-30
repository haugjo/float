import numpy as np
import warnings
from sklearn.metrics import zero_one_loss


def mean_drift_performance_decay(result, known_drifts, batch_size, reference_measure=zero_one_loss, interval=10):
    """
    Performance Decay Under Concept Drift
    Return the average divergence of the reference measure after the start of a known concept drift

    Args:
        result (dict): result dictionary of the prediction evaluator
        known_drifts (list): list of indices indicating positions of known concept drift
        batch_size (int): no. of observations processed per iteration
        reference_measure (function): evaluation measure function
        interval (int): interval after known conept drifts to investigate performance decay

    Returns:
        float: mean performance decay after current (known) concept drifts regarding the reference measure
    """
    init_interval = interval
    len_result = len(result[reference_measure.__name__]['measures'])

    # Get previous mean decay
    if len_result > 0:
        decay = result['mean_drift_performance_decay']['measures'][-1]
    else:
        decay = 0

    # Compute performance decay for each known drift that has already happened
    for i, kd in enumerate(known_drifts):
        if isinstance(kd, tuple):
            drift_t = round(kd[0] / batch_size)  # consider beginning of drift as reference point
        else:
            drift_t = round(kd / batch_size)

        # Check if we are still inside of the interval
        if len_result > drift_t + interval:
            continue
        elif len_result > drift_t:
            # Reset interval if we have not seen enough observations yet
            if len_result < drift_t + interval:
                interval = len_result - drift_t
            if drift_t - interval < 0:
                interval = drift_t

            if reference_measure.__name__ in result:
                before_scores = result[reference_measure.__name__]['measures'][drift_t - interval:drift_t]
                after_scores = result[reference_measure.__name__]['measures'][drift_t:drift_t + interval]
                diff_new = np.mean(after_scores) - np.mean(before_scores)
                decay = decay + (diff_new - decay) / (i * init_interval + interval)  # incremental mean
            else:
                warnings.warn('The reference measure {} is not part of the PredictionEvaluator; we return 0 per '
                              'default. Please provide {} to the PredictionEvaluator and rerun.'.format(
                    reference_measure.__name__, reference_measure.__name__))
        else:
            break  # other known drifts have not been reached yet

    return decay
