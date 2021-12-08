from numpy.typing import Any, ArrayLike
from river.metrics import ClassificationMetric


def river_classification_metric(y_true: ArrayLike,
                                y_pred: ArrayLike,
                                metric: ClassificationMetric,
                                **kwargs) -> Any:
    """
    Acts as a wrapper for river classification methods.

    Args:
        y_true: True target labels.
        y_pred: Predicted target labels.
        metric: The river classification metric.
        kwargs:
            A dictionary containing additional and specific keyword arguments, which are passed to the evaluation
            functions.

    Returns:
        Any: The current value of the specified metric.
    """
    for y_true_i, y_pred_i in zip(y_true, y_pred):
        metric.update(y_true_i, y_pred_i, **kwargs)

    return metric.get()
