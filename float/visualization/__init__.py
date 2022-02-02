"""Visualization Module.

This module contains visualizations that may be used to illustrate the test results of online predictive models,
online feature selection methods and concept drift detection methods. We recommend combining these visualizations with
the float evaluator and pipeline modules to deliver high-quality and standardized experiments.

Copyright (C) 2022 Johannes Haug

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from .plot import plot
from .scatter import scatter
from .bar import bar
from .feature_selection_scatter import feature_selection_scatter
from .feature_selection_bar import feature_selection_bar
from .feature_weight_box import feature_weight_box
from .concept_drift_detection_scatter import concept_drift_detection_scatter
from .spider_chart import spider_chart

__all__ = ['plot', 'scatter', 'bar', 'feature_selection_scatter', 'feature_selection_bar', 'feature_weight_box',
           'concept_drift_detection_scatter', 'spider_chart']
