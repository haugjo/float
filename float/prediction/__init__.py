"""
The float.prediction module includes methods for predicting using defined models.
"""

from .predictor import Predictor
from .sklearn_perceptron import SklearnPerceptron

__all__ = ['Predictor', 'SklearnPerceptron']
