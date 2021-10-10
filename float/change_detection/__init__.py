"""
The float.change_detection module includes methods for concept drift detection.
"""

from .base_change_detector import BaseChangeDetector
from .erics import ERICS

__all__ = ['BaseChangeDetector', 'ERICS']
