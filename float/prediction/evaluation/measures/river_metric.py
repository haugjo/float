"""River Classification Metric Wrapper.

This function is a wrapper for river classification metrics. This wrapper is required, as river metrics cannot process
batches of observations out of the box.

Copyright (C) 2022 Johannes Haug.
"""
from numpy.typing import ArrayLike
from river.metrics import Metric
from typing import Any


def river_metric(y_true: ArrayLike,
                 y_pred: ArrayLike,
                 metric: Metric,
                 **kwargs) -> Any:
    """Wrapper function for river classification metrics.

    Args:
        y_true: True target labels.
        y_pred: Predicted target labels.
        metric: A river object of type Metric.
        kwargs:
            A dictionary containing additional and specific keyword arguments, which are passed to the evaluation
            functions.

    Returns:
        Any: The current value of the specified metric.
    """
    for y_true_i, y_pred_i in zip(y_true, y_pred):
        metric.update(y_true_i, y_pred_i, **kwargs)

    return metric.get()
