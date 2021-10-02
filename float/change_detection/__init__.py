"""
The float.change_detection module includes methods for concept drift detection.
"""

from .base_change_detector import BaseChangeDetector
from .skmultiflow.skmultiflow_change_detector import SkmultiflowChangeDetector
from .erics import ERICS

__all__ = ['BaseChangeDetector', 'SkmultiflowChangeDetector', 'ERICS']
