"""Base Online Predictor Module.

This module encapsulates functionality for online predictive modelling.

Copyright (C) 2022 Johannes Haug.
"""
from .base_predictor import BasePredictor
from .dynamic_model_tree import DynamicModelTreeClassifier

__all__ = ['BasePredictor', 'DynamicModelTreeClassifier']
