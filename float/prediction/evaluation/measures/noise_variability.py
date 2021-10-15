"""Noise Variability Measure.

This function returns the noise variability of a predictor. This measure corresponds to the mean difference of some
performance measure when perturbing the input with noise. It is hence an indication of a predictor's robustness to
noisy inputs.

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
import copy
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import zero_one_loss
from typing import Callable, Optional, List

from float.prediction.base_predictor import BasePredictor


def noise_variability(y_true: ArrayLike, y_pred: ArrayLike, X: ArrayLike, predictor: BasePredictor,
                      reference_measure: Callable = zero_one_loss, cont_noise_loc: float = 0,
                      cont_noise_scale: float = 0.1, cat_features: Optional[list] = None,
                      cat_noise_dist: Optional[List[list]] = None, n_samples: int = 10) -> float:
    """Calculates the variability of a predictor under input noise.

    Args:
        y_true: True target labels.
        y_pred: Predicted target labels.
        X: Array/matrix of observations.
        predictor: Predictor object.
        reference_measure: Evaluation measure function.
        cont_noise_loc: Location (mean) of a normal distribution from which we sample noise for continuous features.
        cont_noise_scale: Scale (variance) of a normal distribution from which we sample noise for continuous features.
        cat_features: List of indices that correspond to categorical features.
        cat_noise_dist: List of lists, where each list contains the noise values of one categorical feature.
        n_samples: Number of times we sample noise and investigate divergence from the original loss.

    Returns:
        float: Mean difference to the original loss for n input perturbations.
    """
    # Init random state Todo: replace with global random state
    rng = np.random.default_rng(0)

    # Get feature indices
    if cat_features is None:
        cat_features = []
    cont_features = np.setdiff1d(np.arange(X.shape[1]), cat_features)

    # Set categorical noise distribution
    if cat_noise_dist is None:
        cat_noise_dist = []
        for cat in cat_features:  # Set categorical noise values (equals unique values in X) if none are specified
            cat_noise_dist.append(np.unique(X[:, cat]))

    divergence = []
    old_score = reference_measure(y_true=y_true, y_pred=y_pred)

    # Perturb input n times and save difference with original measure
    for ns in range(n_samples):
        X_ns = copy.copy(X)

        # Sample noise of continuous features from normal distribution
        cont_noise = rng.normal(loc=cont_noise_loc, scale=cont_noise_scale, size=X_ns[:, cont_features].shape)
        X_ns[:, cont_features] += cont_noise

        if len(cat_features) > 0:
            # Replace categorical feature values with a sampled value from the corresponding list of noise values
            for cf, noise_val in zip(cat_features, cat_noise_dist):
                noise_idx = rng.multinomial(n=1, pvals=[1/len(noise_val)] * len(noise_val), size=X_ns[:, cf].shape)
                noise_idx = np.argmax(noise_idx, axis=1)
                cat_noise = np.array([noise_val[idx] for idx in noise_idx]).reshape(noise_idx.shape)
                X_ns[:, cf] = cat_noise

        # Predict the perturbed input and recompute the evaluation score
        new_pred = predictor.predict(X_ns)
        divergence.append(reference_measure(y_true=y_true, y_pred=new_pred) - old_score)

    return np.mean(divergence).item()
