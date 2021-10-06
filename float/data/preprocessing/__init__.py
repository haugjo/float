"""
The float.data.preprocessing module includes a wrapper for scikit scaler objects and an abstract base class.
"""

from float.data.preprocessing.base_scaler import BaseScaler
from float.data.preprocessing.sklearn_scaler import SklearnScaler

__all__ = ['BaseScaler', 'SklearnScaler']
