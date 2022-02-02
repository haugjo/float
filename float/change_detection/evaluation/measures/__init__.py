"""Evaluation Measures for Concept Drift Detection Models.

This module contains evaluation measures for active/explicit concept drift detection models.

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
from .detected_change_rate import detected_change_rate
from .detection_delay import detection_delay
from .false_discovery_rate import false_discovery_rate
from .mean_time_ratio import mean_time_ratio
from .missed_detection_rate import missed_detection_rate
from .time_between_false_alarms import time_between_false_alarms

__all__ = ['detected_change_rate', 'detection_delay', 'false_discovery_rate', 'mean_time_ratio',
           'missed_detection_rate', 'time_between_false_alarms']
