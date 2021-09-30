"""
The module includes a wrapper for scikit-learn evaluation metrics and custom evaluation measures
"""

from .noise_variability import noise_variability
from .mean_drift_performance_decay import mean_drift_performance_decay

__all__ = ['noise_variability', 'mean_drift_performance_decay']
