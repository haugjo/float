"""Noise Variability Measure.

This function returns the noise variability of a predictor. This measure corresponds to the mean difference of a
performance measure when perturbing the input with noise. It is hence an indication of a predictor's stability under
noisy inputs.

Copyright (C) 2022 Johannes Haug.
"""
import copy
import numpy as np
from numpy.typing import ArrayLike
from numpy.random import Generator
from sklearn.metrics import zero_one_loss
from typing import Callable, Optional, List

from float.prediction.base_predictor import BasePredictor


def noise_variability(y_true: ArrayLike,
                      y_pred: ArrayLike,
                      X: ArrayLike,
                      predictor: BasePredictor,
                      reference_measure: Callable = zero_one_loss,
                      reference_measure_kwargs: Optional[dict] = None,
                      cont_noise_loc: float = 0,
                      cont_noise_scale: float = 0.1,
                      cat_features: Optional[list] = None,
                      cat_noise_dist: Optional[List[list]] = None,
                      n_samples: int = 10,
                      rng: Generator = np.random.default_rng(0)) -> float:
    """Calculates the variability of a predictor under input noise.

    Args:
        y_true: True target labels.
        y_pred: Predicted target labels.
        X: Array/matrix of observations.
        predictor: Predictor object.
        reference_measure: Evaluation measure function.
        reference_measure_kwargs: Keyword arguments of the reference measure.
        cont_noise_loc: Location (mean) of a normal distribution from which we sample noise for continuous features.
        cont_noise_scale: Scale (variance) of a normal distribution from which we sample noise for continuous features.
        cat_features: List of indices that correspond to categorical features.
        cat_noise_dist: List of lists, where each list contains the noise values of one categorical feature.
        n_samples: Number of times we sample noise and investigate divergence from the original loss.
        rng: A numpy random number generator object. The global random state of the pipeline will be used to this end.

    Returns:
        float: Mean difference to the original loss for n input perturbations.
    """
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
    call_args = dict()
    if reference_measure_kwargs is not None:
        for arg, val in reference_measure_kwargs.items():
            call_args[arg] = val
    call_args['y_true'] = y_true
    call_args['y_pred'] = y_pred
    old_score = reference_measure(**call_args)

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
        call_args = dict()
        if reference_measure_kwargs is not None:
            for arg, val in reference_measure_kwargs.items():
                call_args[arg] = val
        call_args['y_true'] = y_true
        call_args['y_pred'] = new_pred
        divergence.append(reference_measure(**call_args) - old_score)

    return np.mean(divergence).item()
