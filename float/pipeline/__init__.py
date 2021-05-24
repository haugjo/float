"""
The float.pipeline module includes methods for creating and triggering events.
"""

from .pipeline import Pipeline
from .prequential_pipeline import PrequentialPipeline
from .holdout_pipeline import HoldoutPipeline

__all__ = ['Pipeline', 'PrequentialPipeline', 'HoldoutPipeline']
