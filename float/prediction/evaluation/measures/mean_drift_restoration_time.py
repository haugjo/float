import numpy as np
import warnings
from sklearn.metrics import zero_one_loss


def mean_drift_restoration_time(result, known_drifts, batch_size, reference_measure=zero_one_loss, incr=False, interval=10):
    """
    Recovery Time After Concept Drift
    Return the average no. of iterations (time steps) before recovering the evaluation measure previous of a drift

    Args:
        result (dict): result dictionary of the prediction evaluator
        known_drifts (list): list of indices indicating positions of known concept drift
        batch_size (int): no. of observations processed per iteration
        reference_measure (function): evaluation measure function
        incr (bool): indicates whether the evaluation measure is incremental (i.e. higher is better)
        interval (int): interval before known concept drift to use as a reference

    Returns:
        float: current mean no. of iterations before recovery from (known) concept drifts
    """
    iter_drifts = reversed(known_drifts)
    len_result = len(result[reference_measure.__name__]['measures'])

    # Get previous mean recovery time
    if len_result > 0:
        recovery = result['mean_drift_restoration_time']['measures'][-1]
    else:
        return 0

    # Identify currently relevant drift
    drift = next(iter_drifts, None)
    i = len(known_drifts)  # drift identifier
    while drift is not None:
        if isinstance(drift, tuple):
            drift = round(drift[0] / batch_size)  # consider beginning of drift as reference point
        else:
            drift = round(drift / batch_size)

        if len_result > drift:
            break
        else:
            i -= 1
            drift = next(iter_drifts, None)

    if drift is not None:
        if len_result > drift + 1 and recovery == result['mean_drift_restoration_time']['measures'][-2]:
            return recovery  # return, if model has already recovered from drift
        else:
            # Get current recovery time
            if reference_measure.__name__ in result:
                value_before = np.mean(result[reference_measure.__name__]['measures'][drift - interval:drift])
                if incr:
                    relevant_idx = np.argwhere(result[reference_measure.__name__]['measures'][drift:] >= value_before).flatten()
                else:
                    relevant_idx = np.argwhere(result[reference_measure.__name__]['measures'][drift:] <= value_before).flatten()

                if len(relevant_idx) > 0:
                    rec_new = relevant_idx[0]
                else:  # model has not recovered yet, use time since drift as return value
                    rec_new = len(result[reference_measure.__name__]['measures'][drift:])

                # Get recovery time prior to current drift
                prev_drift_recovery = result['mean_drift_restoration_time']['measures'][drift - 1]
                recovery = prev_drift_recovery + (rec_new - prev_drift_recovery) / i  # incremental mean
            else:
                warnings.warn('The reference measure {} is not part of the PredictionEvaluator; we return 0 per '
                              'default. Please provide {} to the PredictionEvaluator and rerun.'.format(
                    reference_measure.__name__, reference_measure.__name__))

    return recovery
