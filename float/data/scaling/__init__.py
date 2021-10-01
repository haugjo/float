"""
The float.data.scaling module includes a wrapper for scikit scaler objects and an abstract base class.
"""

from float.data.scaling.base_scaler import BaseScaler
from float.data.scaling.sklearn_scaler import SklearnScaler

__all__ = ['BaseScaler', 'SklearnScaler']
