"""
The module includes a wrapper for scikit-learn evaluation metrics and custom evaluation measures
"""

from .noise_variability import noise_variability
from .mean_drift_performance_deterioration import mean_drift_performance_deterioration
from .mean_drift_restoration_time import mean_drift_restoration_time

__all__ = ['noise_variability', 'mean_drift_performance_deterioration', 'mean_drift_restoration_time']
