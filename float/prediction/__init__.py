"""
The float.prediction module includes methods for predicting using defined models.
"""

from .base_predictor import BasePredictor
from .skmultiflow.skmultiflow_classifier import SkmultiflowClassifier

__all__ = ['BasePredictor', 'SkmultiflowClassifier']
