"""Dynamic Model Tree Classifier.

This module contains the Dynamic Model Tree Classifier proposed in
HAUG, Johannes; BROELEMANN, Klaus; KASNECI, Gjergji. Dynamic Model Tree for Interpretable Data Stream Learning.
In: 38th IEEE International Conference on Data Engineering, DOI: 10.1109/ICDE53745.2022.00237, 2022.

Copyright (C) 2022 Johannes Haug.
"""
from abc import ABCMeta
import copy
import numpy as np
from numpy.typing import ArrayLike
import math
from typing import Optional, Any, List, Tuple
from sklearn.linear_model import SGDClassifier
import warnings

from float.prediction import BasePredictor


class DynamicModelTreeClassifier(BasePredictor):
    """Dynamic Model Tree Classifier.

    This implementation of the DMT uses linear (logit) simple models and the negative log likelihood loss (as described
    in the corresponding paper).

    Attributes:
        classes (List): List of the target classes.
        learning_rate (float): Learning rate of the linear models.
        penalty_term (float): Regularization term for the linear model (0 = no regularization penalty).
        penalty (str):
            String identifier of the type of regularization used by the linear model.
            Either 'l1', 'l2', or 'elasticnet' (see documentation of sklearn SGDClassifier).
        epsilon (float):
            Threshold required before attempting to split or prune based on the Akaike Information Criterion.
            The smaller the epsilon-threshold, the stronger the evidence for splitting/pruning must be. Choose
            0 < epsilon <= 1.
        n_saved_candidates (int): Max. number of saved split candidates per node.
        p_replaceable_candidates (float):
            Max. percent of saved split candidates that can be replaced by new/better candidates per training iteration.
        cat_features (List[int]): List of indices (pos. in the feature vector) corresponding to categorical features.
        root (Node): Root node of the Dynamic Model Tree.
    """
    def __init__(self,
                 classes: List,
                 learning_rate: float = 0.05,
                 penalty_term: float = 0,
                 penalty: str = 'l2',
                 epsilon: float = 10e-8,
                 n_saved_candidates: int = 100,
                 p_replaceable_candidates: float = 0.5,
                 cat_features: Optional[List[int]] = None,
                 reset_after_drift: Optional[bool] = False):
        """Inits the DMT.

        Args:
            classes: List of the target classes.
            learning_rate: Learning rate of the linear models.
            penalty_term: Regularization term for the linear model (0 = no regularization penalty).
            penalty:
                String identifier of the type of regularization used by the linear model.
                Either 'l1', 'l2', or 'elasticnet' (see documentation of sklearn SGDClassifier).
            epsilon:
                Threshold required before attempting to split or prune based on the Akaike Information Criterion.
                The smaller the epsilon-threshold, the stronger the evidence for splitting/pruning must be. Choose
                0 < epsilon <= 1.
            n_saved_candidates: Max. number of saved split candidates per node.
            p_replaceable_candidates:
                Max. percent of saved split candidates that can be replaced by new/better candidates per training iteration.
            cat_features: List of indices (pos. in the feature vector) corresponding to categorical features.
            reset_after_drift:
                A boolean indicating if the predictor will be reset after a drift was detected. Note that the DMT
                automatically adjusts to concept drift and thus generally need not be retrained.
        """
        super().__init__(reset_after_drift=reset_after_drift)

        self.classes = classes
        self.learning_rate = learning_rate
        self.penalty_term = penalty_term
        self.penalty = penalty
        self.epsilon = epsilon
        self.n_saved_candidates = n_saved_candidates
        self.p_replaceable_candidates = p_replaceable_candidates

        if cat_features is None:
            self.cat_features = []
        else:
            self.cat_features = cat_features

        self.root = None

    def partial_fit(self, X: ArrayLike, y: ArrayLike):
        """Updates the predictor.

        Args:
            X: Array/matrix of observations.
            y: Array of corresponding labels.
        """
        if self.root is None:
            self.root = Node(classes=self.classes,
                             n_features=X.shape[1],
                             learning_rate=self.learning_rate,
                             penalty_term=self.penalty_term,
                             penalty=self.penalty,
                             epsilon=self.epsilon,
                             n_saved_candidates=self.n_saved_candidates,
                             p_replaceable_candidates=self.p_replaceable_candidates,
                             cat_features=self.cat_features)

        if np.unique(y)[0] > len(self.classes):
            raise ValueError('DMT: The observed number of classes {} does not match the specified number of '
                             'classes {}.'.format(np.unique(y)[0], len(self.classes)))

        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError('Dynamic Model Trees can only be trained on numeric values. Please encode non-numeric '
                             'features and targets before training a DMT.')

        self.root.update(X=X, y=y)

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predicts the target values.

        Args:
            X: Array/matrix of observations.

        Returns:
            ArrayLike: Predicted labels for all observations.
        """
        if self.root is None:
            raise RuntimeError('DMT has not been fitted before calling predict.')

        y_pred = []
        for x in X:
            y_pred.append(self.root.predict_observation(x=x)[0])

        return np.asarray(y_pred)

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """Predicts the probability of target values.

        Args:
            X: Array/matrix of observations.

        Returns:
            ArrayLike: Predicted probability per class label for all observations.
        """
        if self.root is None:
            raise RuntimeError('DMT has not been fitted before calling predict_proba.')

        y_pred = []
        for x in X:
            y_pred.append(self.root.predict_observation(x=x, get_prob=True)[0])

        return np.asarray(y_pred)

    def reset(self):
        """Resets the predictor."""
        self.root = None
        self.has_been_trained = False

    def n_nodes(self) -> Tuple[int, int, int]:
        """Returns the number of nodes, leaves and the depth of the DMT.

        Returns:
            int: Total number of nodes.
            int: Total number of leaves.
            int: Depth (where a single root node has depth = 1).
        """
        if self.root is None:
            warnings.warn('DMT has not been fitted yet.')
            return 0, 0, 0
        else:
            return self._add_up_node_count(node=self.root)

    def _add_up_node_count(self, node: Any, n_total: int = 0, n_leaf: int = 0, depth: int = 0) -> Tuple[int, int, int]:
        """Recursively computes DMT node counts.

        Args:
            node: Node of the DMT
            n_total: Counter for the total number of nodes.
            n_leaf: Counter for the number of leaf nodes.
            depth: Counter for the tree depth.

        Returns:
            int: Updated number of nodes.
            int: Updated number of leaves.
            int: Updated depth.
        """
        if node.is_leaf:
            return n_total + 1, n_leaf + 1, depth + 1
        else:
            n_total += 1
            depth += 1
            max_depth = depth
            for child in node.children:
                n_total, n_leaf, c_depth = self._add_up_node_count(node=child,
                                                                   n_total=n_total,
                                                                   n_leaf=n_leaf,
                                                                   depth=depth)
                if c_depth > max_depth:
                    max_depth = c_depth

            return n_total, n_leaf, max_depth


# ********************
# Node of the DMT
# ********************
class Node(metaclass=ABCMeta):
    """Node of the Dynamic Model Tree.

    Attributes:
        classes (List): List of the target classes.
        n_features (int): Number of input features.
        learning_rate (float): Learning rate of the linear models.
        penalty_term (float): Regularization term for the linear model (0 = no regularization penalty).
        penalty (str):
            String identifier of the type of regularization used by the linear model.
            Either 'l1', 'l2', or 'elasticnet' (see documentation of sklearn SGDClassifier).
        epsilon (float):
            Threshold required before attempting to split or prune based on the Akaike Information Criterion.
            The smaller the epsilon-threshold, the stronger the evidence for splitting/pruning must be. Choose
            0 < epsilon <= 1.
        n_saved_candidates (int): Max. number of saved split candidates per node.
        p_replaceable_candidates (float):
            Max. percent of saved split candidates that can be replaced by new/better candidates per training iteration.
        cat_features (List[int]): List of indices (pos. in the feature vector) corresponding to categorical features.
        linear_model (Any): Linear (logit) model trained at the node.
        log_likelihood (ArrayLike): Log-likelihood given observations that reached the node.
        counts_left (dict): Number of observations per split candidate falling to the left child.
        log_likelihoods_left (dict): Log-likelihoods of the left child per split candidate.
        gradients_left (dict): Gradients of the left child per split candidate.
        counts_right (dict): Number of observations per split candidate falling to the right child.
        log_likelihoods_right (dict): Log-likelihoods of the left child per split candidate.
        gradients_right (dict): Gradients of the right child per split candidate
        children (List[Node]): List of child nodes.
        split (tuple): Feature/value combination used for splitting.
        is_leaf (bool): Indicator of whether the node is a leaf.
    """
    def __init__(self,
                 classes: List,
                 n_features: int,
                 learning_rate: float,
                 penalty_term: float,
                 penalty: str,
                 epsilon: float,
                 n_saved_candidates: int,
                 p_replaceable_candidates: float,
                 cat_features: List[int]):
        """Inits Node.

        Args:
            classes: List of the target classes.
            n_features: Number of input features.
            learning_rate: Learning rate of the linear models.
            penalty_term: Regularization term for the linear model (0 = no regularization penalty).
            penalty:
                String identifier of the type of regularization used by the linear model.
                Either 'l1', 'l2', or 'elasticnet' (see documentation of sklearn SGDClassifier).
            epsilon:
                Threshold required before attempting to split or prune based on the Akaike Information Criterion.
                The smaller the epsilon-threshold, the stronger the evidence for splitting/pruning must be. Choose
                0 < epsilon <= 1.
            n_saved_candidates: Max. number of saved split candidates per node.
            p_replaceable_candidates:
                Max. percent of saved split candidates that can be replaced by new/better candidates per training iteration.
            cat_features: List of indices (pos. in the feature vector) corresponding to categorical features.
        """
        self.classes = classes
        self.n_classes = len(classes)
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.penalty_term = penalty_term
        self.penalty = penalty
        self.epsilon = epsilon
        self.n_saved_candidates = n_saved_candidates
        self.p_replaceable_candidates = p_replaceable_candidates
        self.cat_features = cat_features

        self.linear_model = SGDClassifier(loss='log',
                                          penalty=penalty,
                                          alpha=penalty_term,
                                          eta0=learning_rate,
                                          learning_rate='constant',
                                          random_state=0)
        self.log_likelihood = 0

        self.counts_left = dict()
        self.log_likelihoods_left = dict()
        self.gradients_left = dict()
        self.counts_right = dict()
        self.log_likelihoods_right = dict()
        self.gradients_right = dict()

        self.children = []
        self.split = None
        self.is_leaf = True

    def update(self, X: ArrayLike, y: ArrayLike):
        """Updates the node and all descendants.

        Update the parameters of the weak model at the given node.
        If the node is an inner node, we attempt to split on a different feature or replace the inner node
        by a leaf and thereby prune all previous children/subbranches. If the node is a leaf node, we attempt to split.

        Args:
            X: Array/matrix of observations.
            y: Array of corresponding labels.
        """
        # Update the simple/linear model
        log_likelihood_X, gradient_X = self._update_linear_model(X, y)

        # Update/add/replace the statistics of candidate splits
        candidate_aic = self._add_and_replace_candidates(X, log_likelihood_X, gradient_X)

        if not self.is_leaf:
            if self.split[0] in self.cat_features:
                left_idx = X[:, self.split[0]] == self.split[1]
                right_idx = X[:, self.split[0]] != self.split[1]
            else:
                left_idx = X[:, self.split[0]] <= self.split[1]
                right_idx = X[:, self.split[0]] > self.split[1]

            X_left, X_right, y_left, y_right = X[left_idx], X[right_idx], y[left_idx], y[right_idx]  # split data

            if len(y_left) > 0:
                self.children[0].update(X=X_left, y=y_left)

            if len(y_right) > 0:
                self.children[1].update(X=X_right, y=y_right)

            do_split, top_split = self._check_for_split(candidate_aic=candidate_aic)

            if not do_split:
                # Replace inner node by leaf
                self.children = []
                self.split = None
                self.is_leaf = True
            elif do_split and self.split != top_split:
                # Replace current split with new top split
                self._make_inner_node(split=top_split)
        else:
            do_split, top_split = self._check_for_split(candidate_aic=candidate_aic)

            if do_split:
                # Replace leaf node by an inner node
                self._make_inner_node(split=top_split)

    def _make_inner_node(self, split: tuple):
        """Makes the node an inner node and creates two child nodes.

        Args:
            split: Feature/value pair of the current optimal split.
        """
        self.split = split
        self.children = []
        self.is_leaf = False

        # Create new children
        for i in range(2):
            self.children.append(Node(classes=self.classes,
                                      n_features=self.n_features,
                                      learning_rate=self.learning_rate,
                                      penalty_term=self.penalty_term,
                                      penalty=self.penalty,
                                      epsilon=self.epsilon,
                                      n_saved_candidates=self.n_saved_candidates,
                                      p_replaceable_candidates=self.p_replaceable_candidates,
                                      cat_features=self.cat_features))

        # Set statistics of LEFT child.
        self.children[0].log_likelihood = self.log_likelihoods_left[split]

        self.children[0].linear_model = copy.deepcopy(self.linear_model)
        self.children[0].linear_model.intercept_ = copy.copy(self.linear_model.intercept_)
        if self.counts_left[split] > 0:  # Apply one gradient update to the parent parameters and assign to child model
            if self.n_classes == 2:
                self.children[0].linear_model.coef_ = copy.deepcopy(self.linear_model.coef_) \
                                                      + self.learning_rate / self.counts_left[split] * self.gradients_left[split][0]  # use any gradient for update
            else:
                self.children[0].linear_model.coef_ = copy.deepcopy(self.linear_model.coef_) \
                                                      + self.learning_rate / self.counts_left[split] * self.gradients_left[split]
        else:
            self.children[0].linear_model.coef_ = copy.deepcopy(self.linear_model.coef_)

        # Set statistics of RIGHT child.
        self.children[1].log_likelihood = self.log_likelihoods_right[split]

        self.children[1].linear_model = copy.deepcopy(self.linear_model)
        self.children[1].linear_model.intercept_ = copy.copy(self.linear_model.intercept_)
        if self.counts_right[split] > 0:  # Apply one gradient update to the parent parameters and assign to child model
            if self.n_classes == 2:
                self.children[1].linear_model.coef_ = copy.deepcopy(self.linear_model.coef_) \
                                                      + self.learning_rate / self.counts_right[split] * self.gradients_right[split][0]
            else:
                self.children[1].linear_model.coef_ = copy.deepcopy(self.linear_model.coef_) \
                                                      + self.learning_rate / self.counts_right[split] * self.gradients_right[split]
        else:
            self.children[1].linear_model.coef_ = copy.deepcopy(self.linear_model.coef_)

    def _update_linear_model(self, X: ArrayLike, y: ArrayLike) -> Tuple[ArrayLike, dict]:
        """Updates the linear model.

        Update the linear model, the log-likelihood and the gradients.

        Args:
            X: Array/matrix of observations.
            y: Array of corresponding labels.

        Returns:
            ArrayLike: Log-likelihoods for current observations.
            dict: Gradients of each class for current observations.
        """
        self.linear_model.partial_fit(X, y, classes=self.classes)

        log_prob_X = self.linear_model.predict_log_proba(X)
        log_likelihood_X = []
        for lp_i, y_i in zip(log_prob_X, y):  # Save probabilities of the true class
            log_likelihood_X.append(lp_i[y_i])
        log_likelihood_X = np.asarray(log_likelihood_X)

        self.log_likelihood += np.sum(log_likelihood_X)

        # Compute Gradients on updated parameters
        gradient_X = dict()
        dot_param_X = np.dot(X, self.linear_model.coef_.T)

        if self.n_classes == 2:  # logit model
            sigmoid = 1 / (1 + np.exp(-dot_param_X))
            gradient_X[0] = (y.reshape(-1, 1) - sigmoid) * X
            gradient_X[1] = gradient_X[0]  # there's just a single gradient in the binary case
        else:  # multinomial logit
            dot_param_X -= np.max(dot_param_X)  # we subtract a large constant to avoid numpy overflow
            sum_exp = np.sum(np.exp(dot_param_X), axis=1)

            for c in range(self.n_classes):
                kron_delta = y == c
                softmax = np.exp(dot_param_X[:, c]) / sum_exp
                gradient_X[c] = X * (kron_delta - softmax).reshape(-1, 1)

        return log_likelihood_X, gradient_X

    def _add_and_replace_candidates(self, X: ArrayLike, log_likelihood_X: ArrayLike, gradient_X: dict) -> dict:
        """Adds and replaces candidate splits and corresponding statistics.

        Identifies the split candidates with highest gain (i.e. smallest AIC) in the given data sample.
        Replaces partition of old candidates, where the current gain of a new candidate exceeds the old gain.
        Adds new candidates if the max. size of saved statistics has not been reached yet.

        Args:
            X: Array/matrix of observations.
            log_likelihood_X: Log-likelihoods for current observations.
            gradient_X: Gradients of each class for current observations.

        Returns:
            dict: AIC of all currently stored split candidates.
        """
        # Update statistics and compute AIC of current candidates
        current_cand_aic = dict()

        for cand in self.log_likelihoods_left.keys():
            if cand[0] in self.cat_features:  # select left child observations
                idx_left = X[:, cand[0]] == cand[1]
            else:
                idx_left = X[:, cand[0]] <= cand[1]

            # Update statistics for potential left and right children
            self.counts_left[cand] += np.count_nonzero(idx_left)
            self.log_likelihoods_left[cand] += np.sum(log_likelihood_X[idx_left])
            self.gradients_left[cand] += np.asarray(
                [np.sum(gradient_X[c][idx_left], axis=0) for c in range(self.n_classes)])

            self.counts_right[cand] += np.count_nonzero(~idx_left)
            self.log_likelihoods_right[cand] += np.sum(log_likelihood_X[~idx_left])
            self.gradients_right[cand] += np.asarray(
                [np.sum(gradient_X[c][~idx_left], axis=0) for c in range(self.n_classes)])

            # Compute AIC
            current_cand_aic[cand], _ = self._aic(cand=cand)

        # Allow replacement of x% of the highest ranking candidates (i.e. candidates with worst AIC)
        replaceable_cand_aic = dict(sorted(current_cand_aic.items(), key=lambda item: item[1])[-math.ceil(
            self.n_saved_candidates * self.p_replaceable_candidates):])

        old_candidates = set(self.log_likelihoods_left.keys())

        for ftr in range(self.n_features):
            if ftr in self.cat_features:
                uniques = np.unique(X[:, ftr])
            else:
                uniques = np.unique(np.trunc(X[:, ftr] * 10 ** 2) / (10 ** 2))  # Trunc. all but the first two decimals

            for val in uniques:
                if (ftr, val) not in old_candidates:  # only replace by candidates that are not already saved
                    if ftr in self.cat_features:
                        idx_left = X[:, ftr] == val
                    else:
                        idx_left = X[:, ftr] <= val

                    aic, cand_stats = self._aic(cand=(ftr, val),
                                                idx_left=idx_left,
                                                log_likelihood_X=log_likelihood_X,
                                                gradient_X=gradient_X)

                    # Find first existing candidate with larger aic
                    replace_cand = next((cand for cand in replaceable_cand_aic.keys()
                                          if replaceable_cand_aic[cand] > aic), None)

                    if replace_cand and len(self.log_likelihoods_left.keys()) == self.n_saved_candidates:
                        del self.counts_left[replace_cand]  # Delete outdated candidate
                        del self.log_likelihoods_left[replace_cand]
                        del self.gradients_left[replace_cand]
                        del self.counts_right[replace_cand]
                        del self.log_likelihoods_right[replace_cand]
                        del self.gradients_right[replace_cand]
                        del current_cand_aic[replace_cand]
                        del replaceable_cand_aic[replace_cand]

                    # Add new candidate as long as max size is not reached
                    if len(self.log_likelihoods_left.keys()) < self.n_saved_candidates:
                        self.counts_left[(ftr, val)] = cand_stats['count_left']
                        self.log_likelihoods_left[(ftr, val)] = cand_stats['likelihood_left']
                        self.gradients_left[(ftr, val)] = cand_stats['gradient_left']
                        self.counts_right[(ftr, val)] = cand_stats['count_right']
                        self.log_likelihoods_right[(ftr, val)] = cand_stats['likelihood_right']
                        self.gradients_right[(ftr, val)] = cand_stats['gradient_right']

                        current_cand_aic[(ftr, val)] = aic
                        replaceable_cand_aic[(ftr, val)] = aic

        return current_cand_aic

    def _check_for_split(self, candidate_aic: dict) -> Tuple[bool, Optional[tuple]]:
        """Checks if we need to split the node.

        Identifies the split candidate with top gain and checks whether there is enough evidence to split.

        Args:
            candidate_aic: AIC of all currently stored split candidates.

        Returns:
            bool: Indicator whether to do a split or not.
            tuple: Top feature/value pair that should be used for splitting.
        """
        # Get best split candidate with minimal AIC
        cand = min(candidate_aic, key=candidate_aic.get)
        aic_cand = candidate_aic[cand]

        # AIC for a leaf node
        k = self.n_features * self.n_classes if self.n_classes > 2 else self.n_features  # no. of free parameters
        aic_leaf = 2 * k - 2 * self.log_likelihood

        # Perform a statistical test based on the corrected Akaike Information Criterion
        if self.is_leaf:
            # Test at leaf node
            if aic_cand < aic_leaf and math.exp((aic_cand - aic_leaf) / 2) <= self.epsilon:
                return True, cand  # make split
            else:
                return False, cand  # make no split
        else:
            # AIC of the subtree
            log_like_subtree, leaf_count = Node._sum_leaf_likelihoods(self)
            k = leaf_count * self.n_features * self.n_classes if self.n_classes > 2 else leaf_count * self.n_features
            aic_subtree = 2 * k - 2 * log_like_subtree

            # Test at inner node
            if aic_leaf < aic_subtree and math.exp((aic_leaf - aic_subtree) / 2) <= self.epsilon:
                return False, None  # Prune and make leaf
            elif aic_cand < aic_subtree and math.exp((aic_cand - aic_subtree) / 2) <= self.epsilon:
                return True, cand  # Prune and replace with another candidate
            else:
                return True, self.split  # Retain current split

    def _aic(self,
             cand: tuple,
             idx_left: Optional[ArrayLike] = None,
             log_likelihood_X: Optional[ArrayLike] = None,
             gradient_X: Optional[ArrayLike] = None) -> Tuple[float, Optional[dict]]:
        """Computes the Akaike Information Criterion of a given split candidate.

        Args:
            cand: Split candidate (feature/value combination) for which we compute the AIC.
            idx_left: Bool-array indicating all current observations that fall to the left child
            log_likelihood_X: Log-likelihoods for current observations.
            gradient_X: Gradients of each class for current observations.

        Returns:
            float: AIC score for the given split candidate.
            Optional[dict]: statistics (count, likelihood, gradient) of the new split candidate.
        """
        log_like = 0
        new_cand_stats = None

        if cand in self.counts_left:  # If candidate is already stored
            for count, likelihood, gradient in zip([self.counts_left[cand], self.counts_right[cand]],
                                                   [self.log_likelihoods_left[cand], self.log_likelihoods_right[cand]],
                                                   [self.gradients_left[cand], self.gradients_right[cand]]):
                if count > 0:
                    xmax = np.max(np.abs(gradient), axis=1)  # we normalize the gradients to avoid numpy overflow
                    xmax[xmax == 0] = 1  # replace xmax 0 by 1 to avoid division by zero
                    norm = np.linalg.norm(gradient / xmax.reshape(-1, 1), axis=1) * xmax
                    log_like += likelihood + np.sum((self.learning_rate / count) * norm ** 2)

        else:  # If a candidate is not already stored, we need to approximate the likelihood based on current sample.
            # As we replace split candidates over time, new candidate likelihoods might not correspond to the same no.
            # of observations as the parent node. To compare AICs, we scale the candidate likelihoods according
            # to the corresponding stats of the parent node.
            relative_frac = np.abs(np.divide(np.sum(log_likelihood_X[idx_left]),
                                             np.sum(log_likelihood_X[idx_left]) + np.sum(log_likelihood_X[~idx_left]),
                                             out=np.zeros_like(self.log_likelihood),
                                             where=(np.sum(log_likelihood_X[idx_left]) + np.sum(log_likelihood_X[~idx_left])) != 0
                                             )
                                   )
            likelihood_left = (self.log_likelihood - np.sum(log_likelihood_X)) * relative_frac + np.sum(log_likelihood_X[idx_left])
            likelihood_right = (self.log_likelihood - np.sum(log_likelihood_X)) * (1 - relative_frac) + np.sum(log_likelihood_X[~idx_left])

            count_left = np.count_nonzero(idx_left)
            count_right = np.count_nonzero(~idx_left)
            gradient_left = np.asarray([np.sum(gradient_X[c][idx_left], axis=0) for c in range(self.n_classes)])
            gradient_right = np.asarray([np.sum(gradient_X[c][~idx_left], axis=0) for c in range(self.n_classes)])

            new_cand_stats = {
                'count_left': count_left,
                'count_right': count_right,
                'likelihood_left': 0,
                'likelihood_right': 0,
                'gradient_left': gradient_left,
                'gradient_right': gradient_right
            }

            for like_id, count, likelihood, gradient in zip(['likelihood_left', 'likelihood_right'],
                                                            [count_left, count_right],
                                                            [likelihood_left, likelihood_right],
                                                            [gradient_left, gradient_right]):
                if count > 0:
                    xmax = np.max(np.abs(gradient), axis=1)  # we normalize the gradients to avoid numpy overflow
                    xmax[xmax == 0] = 1  # replace xmax 0 by 1 to avoid division by zero
                    norm = np.linalg.norm(gradient / xmax.reshape(-1, 1), axis=1) * xmax
                    new_cand_stats[like_id] = likelihood + np.sum((self.learning_rate / count) * norm ** 2)
                else:
                    new_cand_stats[like_id] = likelihood

                log_like += new_cand_stats[like_id]

        # Calculate no. of free parameters (i.e., parameters for a left and right child node)
        k = 2 * self.n_features * self.n_classes if self.n_classes > 2 else 2 * self.n_features
        aic = 2 * k - 2 * log_like

        return aic, new_cand_stats

    @staticmethod
    def _sum_leaf_likelihoods(node: Any, likelihood: int = 0, leaf_count: int = 0) -> Tuple[int, int]:
        """Sums up the likelihoods at the leaves of a subtree.

        Args:
            node: Current node in the DMT.
            likelihood: Sum of laef likelihoods.
            leaf_count: Counter of leaf nodes.

        Returns:
            float: Updated sum of likelihoods.
            int: Updated counter of leaf nodes.
        """
        if node.is_leaf:
            return likelihood + node.log_likelihood, leaf_count + 1
        else:
            likelihood, leaf_count = Node._sum_leaf_likelihoods(node.children[0], likelihood, leaf_count)
            likelihood, leaf_count = Node._sum_leaf_likelihoods(node.children[1], likelihood, leaf_count)
            return likelihood, leaf_count

    def predict_observation(self, x: ArrayLike, get_prob: bool = False) -> ArrayLike:
        """Predicts one observation (recurrent function).

        Passes an observation down the tree until a leaf is reached. Makes prediction at leaf.

        Args:
            x: Observation.
            get_prob: Indicator whether to return class probabilities.

        Returns:
            ArrayLike: Predicted class label/probability of the given observation.
        """
        if self.is_leaf:
            x = x.reshape(1, -1)
            if get_prob:
                return self.linear_model.predict_proba(x)
            else:
                return self.linear_model.predict(x)
        else:
            if self.split[0] in self.cat_features:
                if x[self.split[0]] == self.split[1]:  # advance to children
                    return self.children[0].predict_observation(x)
                else:
                    return self.children[1].predict_observation(x)
            else:
                if x[self.split[0]] <= self.split[1]:  # advance to children
                    return self.children[0].predict_observation(x)
                else:
                    return self.children[1].predict_observation(x)

