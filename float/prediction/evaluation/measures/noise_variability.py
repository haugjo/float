import numpy as np
import copy
from sklearn.metrics import zero_one_loss


def noise_variability(y_true, y_pred, X, predictor, reference_measure=zero_one_loss, cont_noise_loc=0,
                      cont_noise_scale=0.1, cat_features=None, cat_noise_dist=None, n_samples=10):
    """
    Variability Under Input Noise (as indication of algorithmic stability).
    Return the mean divergence (difference) from the original loss for n input perturbations.

    Args:
        y_true (list | np.array): true target label
        y_pred (list | np.array): predicted target label
        X (np.array): matrix of observations
        reference_measure (function): evaluation measure function
        predictor (BasePredictor): predictor object
        cont_noise_loc (float): location (mean) of normal distribution from which to sample noise for continuous features
        cont_noise_scale (float): scale (variance) of normal distribution from which to sample noise for continuous features
        cat_features (list | None): list of indices that correspond to categorical features
        cat_noise_dist (list | None): list of lists, where each list contains the noise values of one categorical feature
        n_samples (int): no. of times to sample noise and investigate divergence from original loss

    Returns:
        float: the mean difference to the original loss for n input perturbations
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

    return np.mean(divergence)
