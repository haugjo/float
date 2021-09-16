"""
The float.change_detection module includes methods for concept drift detection.
"""

from .base_change_detector import BaseChangeDetector
from .skmultiflow_drift_detector import SkmultiflowDriftDetector
from .erics import ERICS

__all__ = ['BaseChangeDetector', 'SkmultiflowDriftDetector', 'ERICS']
