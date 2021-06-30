"""
The float.prediction module includes methods for predicting using defined models.
"""

from .predictor import Predictor
from .skmultiflow_perceptron import SkmultiflowPerceptron

__all__ = ['Predictor', 'SkmultiflowPerceptron']
