"""
The float.evaluation module includes evaluation methods for online learning.
"""

from .evaluator import Evaluator
from .time_metric import TimeMetric
from .change_detection_metric import ChangeDetectionMetric

__all__ = ['Evaluator', 'TimeMetric', 'ChangeDetectionMetric']
