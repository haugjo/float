"""
The float.feature_selector module includes online feature selection methods.
"""

from .base_feature_selector import BaseFeatureSelector
from .cancel_out import CancelOutFeatureSelector
from .efs import EFS
from .fires import FIRES
from .fsds import FSDS
from .ofs import OFS

__all__ = ['BaseFeatureSelector', 'CancelOutFeatureSelector', 'EFS', 'FIRES', 'FSDS', 'OFS']
