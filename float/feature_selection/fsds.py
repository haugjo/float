from float.feature_selection.base_feature_selector import BaseFeatureSelector
import numpy as np
import numpy.linalg as ln


class FSDS(BaseFeatureSelector):
    """
    Feature Selection on Data Streams.

    Based on a paper by Huang et al. (2015). Feature Selection for unsupervised Learning.
    This code is copied from the Python implementation of the authors with minor reductions and adaptations.
    """
    def __init__(self, n_total_features, n_selected_features, l=0, m=None, B=None, k=2, reset_after_drift=False,
                 baseline='constant', ref_sample=0):
        """
        Initializes the FSDS feature selector.

        Args:
            n_total_features (int): total number of features
            n_selected_features (int): number of selected features
            l (int): size of the matrix sketch with l << m
            m (int): size of the feature space
            B (list/np.ndarray): matrix sketch
            k (int): number of singular vectors with k <= ell
            reset_after_drift (bool): indicates whether to reset the predictor after a drift was detected
            baseline (str): identifier of baseline method (value to replace non-selected features with)
            ref_sample (float | np.array): integer (in case of 'constant' baseline) or sample used to obtain the baseline
        """
        super().__init__(n_total_features, n_selected_features, supports_multi_class=False,
                         reset_after_drift=reset_after_drift, baseline=baseline, ref_sample=ref_sample)

        self.m_init = m
        self.B_init = B
        self.m = n_total_features if m is None else m
        self.B = [] if B is None else B
        self.l = l
        self.k = k

    def weight_features(self, X, y):
        """
        Given a batch of observations and corresponding labels, computes feature weights.

        Args:
            X (np.ndarray): samples of current batch
            y (np.ndarray): labels of current batch
        """
        Yt = X.T  # algorithm assumes rows to represent features

        if self.l < 1:
            self.l = int(np.sqrt(self.m))

        if len(self.B) == 0:
            # for Y0, we need to first create an initial sketched matrix
            self.B = Yt[:, :self.l]
            C = np.hstack((self.B, Yt[:, self.l:]))
            n = Yt.shape[1] - self.l
        else:
            # combine current sketched matrix with input at time t
            # C: m-by-(n+ell) matrix
            C = np.hstack((self.B, Yt))
            n = Yt.shape[1]

        U, s, V = ln.svd(C, full_matrices=False)
        U = U[:, :self.l]
        s = s[:self.l]
        V = V[:, :self.l]

        # shrink step in Frequent Directions algorithm
        # (shrink singular values based on the squared smallest singular value)
        delta = s[-1] ** 2
        s = np.sqrt(s ** 2 - delta)

        # -- Extension of original code --
        # replace nan values with 0 to prevent division by zero error for small batch numbers
        s = np.nan_to_num(s)

        # update sketched matrix B
        # (focus on column singular vectors)
        self.B = np.dot(U, np.diag(s))

        # According to Section 5.1, for all experiments,
        # the authors set alpha = 2^3 * sigma_k based on the pre-experiment
        alpha = (2 ** 3) * s[self.k - 1]

        # solve the ridge regression by using the top-k singular values
        # X: m-by-k matrix (k <= ell)
        D = np.diag(s[:self.k] / (s[:self.k] ** 2 + alpha))

        # -- Extension of original code --
        # replace nan values with 0 to prevent division by zero error for small batch numbers
        D = np.nan_to_num(D)

        X = np.dot(U[:, :self.k], D)

        self.raw_weight_vector = np.amax(abs(X), axis=1)

    def reset(self):
        """
         Reset matrix sketch
        """
        self.m = self.n_total_features if self.m_init is None else self.m_init
        self.B = [] if self.B_init is None else self.B_init
