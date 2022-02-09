"""Online Feature Selection Module.

This module encapsulates functionality for online feature weighting and selection.

Copyright (C) 2022 Johannes Haug.
"""
from .base_feature_selector import BaseFeatureSelector
from .cancel_out import CancelOutFeatureSelector
from .efs import EFS
from .fires import FIRES
from .fsds import FSDS
from .ofs import OFS

__all__ = ['BaseFeatureSelector', 'CancelOutFeatureSelector', 'EFS', 'FIRES', 'FSDS', 'OFS']
