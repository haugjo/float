"""
A modular evaluation package built on the online learning framework scikit-multiflow.
"""

from . import concept_drift_detection
from . import data
from . import evaluation
from . import event
from . import feature_selection
from . import pipeline
from . import prediction
from . import testing
from . import visualization

__all__ = ['concept_drift_detection', 'data', 'evaluation', 'event', 'feature_selection', 'pipeline', 'prediction',
           'testing', 'visualization']
