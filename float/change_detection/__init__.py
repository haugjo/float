"""Change Detection Module.

This module encapsulates functionality for global and partial (i.e. feature-wise) concept drift detection, as well as
implementations of promising concept drift detection methods.

Copyright (C) 2022 Johannes Haug.
"""
from .base_change_detector import BaseChangeDetector
from .erics import ERICS

__all__ = ['BaseChangeDetector', 'ERICS']
