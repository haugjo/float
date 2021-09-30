"""
The module includes a wrapper for scikit-learn evaluation metrics and custom evaluation measures
"""

from .noise_variability import noise_variability
from .mean_drift_performance_decay import mean_drift_performance_decay
from .mean_drift_recovery_time import mean_drift_recovery_time

__all__ = ['noise_variability', 'mean_drift_performance_decay', 'mean_drift_recovery_time']
