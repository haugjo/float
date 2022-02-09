"""Pipeline Module.

This module contains functionality to construct a pipeline and run experiments in a standardized and modular fashion.

Copyright (C) 2022 Johannes Haug.
"""
from .base_pipeline import BasePipeline
from .distributed_fold_pipeline import DistributedFoldPipeline
from .holdout_pipeline import HoldoutPipeline
from .prequential_pipeline import PrequentialPipeline

__all__ = ['BasePipeline', 'DistributedFoldPipeline', 'HoldoutPipeline', 'PrequentialPipeline']
