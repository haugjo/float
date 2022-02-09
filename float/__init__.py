"""Frictionless Online Analysis and Testing (float).

Float is a modular Python framework for standardised evaluation of online learning models.

Copyright (C) 2022 Johannes Haug.
"""
from . import change_detection
from . import data
from . import feature_selection
from . import pipeline
from . import prediction
from . import visualization

__all__ = ['change_detection', 'data', 'feature_selection', 'pipeline', 'prediction', 'visualization']
