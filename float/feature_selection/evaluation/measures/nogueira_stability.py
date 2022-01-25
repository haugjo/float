"""Nogueira Feature Set Stability Measure.

This function returns the feature set stability measure introduced by:
NOGUEIRA, Sarah; SECHIDIS, Konstantinos; BROWN, Gavin. On the stability of feature selection algorithms.
J. Mach. Learn. Res., 2017, 18. Jg., Nr. 1, S. 6345-6398.

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
from typing import List


def nogueira_stability(selected_features_history: List[list], n_total_features: int) -> float:
    """Calculates the Nogueira measure for feature selection stability.

    Args:
        selected_features_history: A list of all selected feature vectors obtained over time.
        n_total_features: The total number of features.

    Returns:
        float: The feature set stability due to Nogueira et al.
    """
    Z = np.zeros([min(len(selected_features_history), 2), n_total_features])
    for row, col in enumerate(selected_features_history[-2:]):  # Compare most currently selected feature sets.
        Z[row, col] = 1

    try:
        M, d = Z.shape
        hatPF = np.mean(Z, axis=0)
        kbar = np.sum(hatPF)
        denom = (kbar / d) * (1 - kbar / d)
        stability = 1 - (M / (M - 1)) * np.mean(np.multiply(hatPF, 1 - hatPF)) / denom
    except ZeroDivisionError:
        stability = 0  # The measure requires at least 2 measurements and thus runs an error at t=1

    return stability
