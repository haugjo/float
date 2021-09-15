"""
The float.concept_drift_detection.tornado module includes concept drift detection methods from the tornado package.
"""

from .adwin import Adwin
from .page_hinkley import PageHinkley

__all__ = ['Adwin', 'PageHinkley']