"""
The float.concept_drift_detection module includes methods for concept drift detection.
"""

from .concept_drift_detector import ConceptDriftDetector
from .skmultiflow_drift_detector import SkmultiflowDriftDetector
from .erics import ERICS

__all__ = ['ConceptDriftDetector', 'SkmultiflowDriftDetector', 'ERICS']
