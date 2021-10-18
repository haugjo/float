"""Detection Delay Measure.

This function returns the average delay in number of observations between the beginning of a known concept drift
and the first detected concept drift.

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
import numpy as np
from typing import Optional

from float.change_detection.evaluation.change_detection_evaluator import ChangeDetectionEvaluator


def detection_delay(evaluator: ChangeDetectionEvaluator, drifts: list, n_delay: Optional[int] = None) -> float:
    """Calculates the average delay before detecting a concept drift.

    Args:
        evaluator: The ChangeDetectionEvaluator object.
        drifts: List of time steps corresponding to detected concept drifts.
        n_delay: This attribute is only included for consistency purposes. It is not relevant for this measure.

    Returns:
        float: The average delay in number of observations between a known drift and the first detected drift.
    """
    iter_drifts = iter(evaluator.known_drifts)
    detections = np.asarray(drifts) * evaluator.batch_size  # Translate drifts to relative position in dataset
    delay = 0
    drift = next(iter_drifts, None)
    while drift is not None:
        # Find start of known drift
        if isinstance(drift, tuple):  # Incremental/gradual drifts involve a starting and end point
            start_search = drift[0]
        else:
            start_search = drift

        # Find end of considered search space
        drift = next(iter_drifts, None)
        if drift is not None:
            if isinstance(drift, tuple):  # End of search space = start of next known drift
                end_search = drift[0]
            else:
                end_search = drift
        else:
            end_search = evaluator.n_total  # If no more known concept drift, set end search space to total no. of obs.

        # Find first relevant drift detection
        relevant_drift = next((det for det in detections if start_search <= det < end_search), None)

        if relevant_drift is not None:
            delay += relevant_drift - start_search
        else:
            delay += end_search - start_search  # Add a complete period as default delay

    return delay / len(evaluator.known_drifts)
