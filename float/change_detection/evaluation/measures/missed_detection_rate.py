"""Missed Detection Rate Measure.

This function returns the rate of known concept drifts that the model has failed to detect.
The measure was introduced by
Bifet, Albert, et al. "CD-MOA: Change detection framework for massive online analysis."
International Symposium on Intelligent Data Analysis. Springer, Berlin, Heidelberg, 2013.

The missed detection rate is equal to 1 - detected_change_rate.

Copyright (C) 2021 Johannes Haug

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
from float.change_detection.evaluation.measures.detected_change_rate import detected_change_rate
from float.change_detection.evaluation.change_detection_evaluator import ChangeDetectionEvaluator


def missed_detection_rate(evaluator: ChangeDetectionEvaluator, drifts: list, n_delay: int) -> float:
    """Calculates the rate of missed known concept drifts.

    Args:
        evaluator: The ChangeDetectionEvaluator object.
        drifts: List of time steps corresponding to detected concept drifts.
        n_delay:
            The number of observations after a known concept drift, during which we count the detections made by the
            model as true positives.

    Returns:
        float: The fraction of missed known concept drifts.
    """
    return 1 - detected_change_rate(evaluator=evaluator, drifts=drifts, n_delay=n_delay)
