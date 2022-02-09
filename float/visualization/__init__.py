"""Visualization Module.

This module contains visualizations that can be used to illustrate the test results of a float pipeline evaluation.
Specifically, the visualization module provides custom versions of standard plot types like line, scatter or bar, as
well as dedicated plot types for the illustration and comparison of online predictive models, online feature selection
methods and concept drift detection methods.

Copyright (C) 2022 Johannes Haug.
"""
from .plot import plot
from .scatter import scatter
from .bar import bar
from .feature_selection_scatter import feature_selection_scatter
from .feature_selection_bar import feature_selection_bar
from .feature_weight_box import feature_weight_box
from .change_detection_scatter import change_detection_scatter
from .spider_chart import spider_chart

__all__ = ['plot', 'scatter', 'bar', 'feature_selection_scatter', 'feature_selection_bar', 'feature_weight_box',
           'change_detection_scatter', 'spider_chart']
