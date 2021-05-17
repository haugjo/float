"""
The float.evaluation module includes evaluation methods for online learning.
"""

from .evaluator import Evaluator
from .time_metric import TimeMetric

__all__ = ['Evaluator', 'TimeMetric']
