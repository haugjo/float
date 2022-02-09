"""Preprocessing Module.

This module encapsulates functionality to scale, i.e. normalize, streaming observations.

Copyright (C) 2022 Johannes Haug.
"""
from float.data.preprocessing.base_scaler import BaseScaler
from float.data.preprocessing.sklearn_scaler import SklearnScaler

__all__ = ['BaseScaler', 'SklearnScaler']
