"""
A modular evaluation package built on the online learning framework scikit-multiflow.
"""

from . import change_detection
from . import data
from . import feature_selection
from . import pipeline
from . import prediction
from . import unit_tests
from . import visualization

__all__ = ['change_detection', 'data', 'feature_selection', 'pipeline', 'prediction',
           'unit_tests', 'visualization']
