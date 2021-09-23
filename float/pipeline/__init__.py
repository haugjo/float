"""
The float.pipeline module includes methods for creating and triggering events.
"""

from .base_pipeline import BasePipeline
from .prequential_pipeline import PrequentialPipeline
from .holdout_pipeline import HoldoutPipeline

__all__ = ['BasePipeline', 'PrequentialPipeline', 'HoldoutPipeline']
