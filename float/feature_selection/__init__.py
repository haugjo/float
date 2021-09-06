"""
The float.feature_selector module includes online feature selection methods.
"""

from .feature_selector import FeatureSelector
from .cancel_out import CancelOutFeatureSelector
from .efs import EFS
from .fires import FIRES
from .fsds import FSDS
from .ofs import OFS

__all__ = ['FeatureSelector', 'CancelOutFeatureSelector', 'EFS', 'FIRES', 'FSDS', 'OFS']
