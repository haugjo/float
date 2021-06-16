"""
The float.feature_selector module includes online feature selection methods.
"""

from .feature_selector import FeatureSelector
from .cancel_out import CancelOut
from .efs import EFS
from .fires import FIRES
from .fsds import FSDS
from .ofs import OFS

__all__ = ['FeatureSelector', 'CancelOut', 'EFS', 'FIRES', 'FSDS', 'OFS']
