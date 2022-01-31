"""River Feature Selection Model Wrapper.

This module contains a wrapper for the river feature selection models.

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
from numpy.random import Generator
from numpy.typing import ArrayLike
from river.base import Transformer
from river.feature_selection.k_best import SelectKBest
from river.feature_selection.random import PoissonInclusion
from river.feature_selection.variance import VarianceThreshold
from typing import Union, List

from float.feature_selection import BaseFeatureSelector


class RiverFeatureSelector(BaseFeatureSelector):
    """Wrapper for river feature selection models."""
    def __init__(self,
                 model: Transformer,
                 feature_names: List[str],
                 n_total_features: int,
                 reset_after_drift: bool = False,
                 baseline: str = 'constant',
                 ref_sample: Union[float, ArrayLike] = 0):
        """Inits the feature selector.

        Args:
            model: The river feature selector object (one of SelectKBest, PoissonInclusion or VarianceThreshold).
            feature_names: A list of all feature names.
            n_total_features: See description of base class.
            reset_after_drift: See description of base class.
            baseline: See description of base class.
            ref_sample: See description of base class.
        """
        self.init_model = model.clone()
        self.model = model
        self.feature_names = feature_names
        super().__init__(n_total_features=n_total_features,
                         n_selected_features=self.model.k if isinstance(self.model, SelectKBest) else None,
                         supports_multi_class=False,
                         reset_after_drift=reset_after_drift,
                         baseline=baseline,
                         ref_sample=ref_sample)

    def weight_features(self, X: ArrayLike, y: ArrayLike):
        """Updates feature weights."""
        # PoissonInclusion uses random selection so weights do not need to be set.
        if not isinstance(self.model, PoissonInclusion):
            for x, y_i in zip(X, y):
                x = {key: value for key, value in zip(self.feature_names, x)}

                if isinstance(self.model, VarianceThreshold):
                    self.model.learn_one(x=x)
                else:
                    self.model.learn_one(x=x, y=bool(y_i))

            if isinstance(self.model, SelectKBest):
                self.weights = np.array(list(self.model.leaderboard.values()))
            else:
                self.weights = np.array([var.get() for var in self.model.variances.values()])

    def select_features(self, X: ArrayLike, rng: Generator) -> ArrayLike:
        """Selects features with highest absolute weights.

        This overrides the corresponding parent class function.
        """
        selected_features = []
        # SelectKBest can be used with the superclass function as it selects a fixed number of features
        if isinstance(self.model, SelectKBest):
            return super().select_features(X, rng)

        X_new = self._get_baseline(X=X, rng=rng)

        # VarianceThreshold selects a variable number of features so explicit handling is necessary
        if isinstance(self.model, VarianceThreshold):
            x_dict = {key: value for key, value in zip(self.feature_names, X[0])}
            # Features are selected from the keys of the return value of the transform_one function
            # (same result for all samples until the FS is trained again),
            selected_features = np.where(np.isin(self.feature_names,
                                                 list(self.model.transform_one(x_dict).keys())))[0]
            X_new[:, selected_features] = X[:, selected_features]

        # PoissonInclusion selects random features for each sample
        elif isinstance(self.model, PoissonInclusion):
            self.weights = np.zeros(X.shape[1])
            for i in range(len(X)):
                x_dict = {key: value for key, value in zip(self.feature_names, X[i])}
                selected_features = np.where(np.isin(self.feature_names,
                                                     list(self.model.transform_one(x_dict).keys())))[0]
                # Weights are set to 1 for selected and 0 for non-selected feature.
                # We average them w.r.t the batch size.
                self.weights += np.array([1 if i in selected_features else 0 for i in range(len(self.feature_names))])
                X_new[i, selected_features] = X[i, selected_features]
            self.weights /= len(X)

        self.selected_features = selected_features
        self.weights_history.append(self.weights)
        self.selected_features_history.append(selected_features)

        return X_new

    def reset(self):
        """Resets the feature selector."""
        self.model = copy.deepcopy(self.init_model)
        np.zeros(self.n_total_features)
