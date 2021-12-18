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
import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike
from river.base import Transformer
from river.feature_selection.variance import VarianceThreshold
from typing import Union, List

from float.feature_selection import BaseFeatureSelector


class RiverFeatureSelector(BaseFeatureSelector):
    """Wrapper for river feature selection models."""
    def __init__(self,
                 feature_selector: Transformer,
                 feature_names: List[str],
                 n_total_features: int,
                 reset_after_drift: bool = False,
                 baseline: str = 'constant',
                 ref_sample: Union[float, ArrayLike] = 0):
        """Inits the feature selector.

        Args:
            feature_selector: The river feature selector object.
            feature_names: A list of all feature names.
            n_total_features: See description of base class.
            reset_after_drift: See description of base class.
            baseline: See description of base class.
            ref_sample: See description of base class.
        """
        self.init_feature_selector = feature_selector.clone()
        self.feature_selector = feature_selector
        self.feature_names = feature_names
        super().__init__(n_total_features=n_total_features,
                         n_selected_features=None,  # the number of selected features must be passed to the Transformer object directly (if applicable)
                         supports_multi_class=False,
                         reset_after_drift=reset_after_drift,
                         baseline=baseline,
                         ref_sample=ref_sample)

    def weight_features(self, X: ArrayLike, y: ArrayLike):
        # TODO: set weights and weights_history
        for x, y_i in zip(X, y):
            x = {key: value for key, value in zip(self.feature_names, x)}
            self.feature_selector.learn_one(x=x) if type(self.feature_selector) is VarianceThreshold else self.feature_selector.learn_one(x=x, y=bool(y_i))

    def select_features(self, X: ArrayLike, rng: Generator) -> ArrayLike:
        selected_features = []
        X_new = self._get_baseline(X=X, rng=rng)
        for i in range(len(X)):
            x_dict = {key: value for key, value in zip(self.feature_names, X[i])}
            selected_features = np.where(np.isin(self.feature_names, list(self.feature_selector.transform_one(x_dict).keys())))[0]
            X_new[i, selected_features] = X[i, selected_features]

        self.selected_features = selected_features
        self.selected_features_history.append(selected_features)
        X_new = self._get_baseline(X=X, rng=rng)
        X_new[:, self.selected_features] = X[:, self.selected_features]

        return X_new

    def reset(self):
        self.__init__(feature_selector=self.init_feature_selector,
                      feature_names=self.feature_names,
                      n_total_features=self.n_total_features,
                      reset_after_drift=self.reset_after_drift,
                      baseline=self.baseline,
                      ref_sample=self.ref_sample)
