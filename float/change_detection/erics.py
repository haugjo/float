import numpy as np
from scipy.stats import norm
from warnings import warn
import copy
from float.change_detection.base_change_detector import BaseChangeDetector


class ERICS(BaseChangeDetector):
    """
    ERICS: Effective and Robust Identification of Concept Shift
    please cite:
    [1] Haug, Johannes and Kasneci, Gjergji. "Learning Parameter Distributions to Detect Concept Drift in Data Streams".
    CoRR. 2020.
    [2] Haug, Johannes, et al. "Leveraging Model Inherent Variable Importance for Stable Online Feature Selection".
    Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020.
    """
    def __init__(self, n_param, window_mvg_average=50, window_drift_detect=50, beta=0.0001,
                 base_model='probit', init_mu=0, init_sigma=1, epochs=10, lr_mu=0.01, lr_sigma=0.01):
        """
        Initializes the ERICS concept drift detector.

        Args:
            n_param (int): total no. of parameters (corresponds to no. of features for probit model)
            window_mvg_average (int): window Size for computation of moving average
            window_drift_detect (int): window Size for Drift Detection
            beta (float): update rate for the alpha-threshold
            base_model (str): name of the base predictive model (whose parameters we investigate)
            init_mu (int): initialize mean of parameter distributions (according to [2])
            init_sigma (int): initialize variance of parameter distributions (according to [2])
            epochs (int): number of epochs for optimization of parameter distributions (according to [2])
            lr_mu (float): learning rate for the gradient update of the mean (according to [2])
            lr_sigma (float): learning rate for the gradient update of the variance (according to [2])
        """
        # Todo: should we allow to reset_after_drift for consistency and overwrite with warning?
        super().__init__(reset_after_drift=False, error_based=False)
        # User-set ERICS-hyperparameters
        self.n_param = n_param
        self.M = window_mvg_average
        self.W = window_drift_detect
        self.beta = beta
        self.base_model = base_model

        # Default hyperparameters
        self.time_step = 0                                           # Current Time Step
        self.time_since_last_global_drift = 0                        # Time steps since last global drift detection
        self.time_since_last_partial_drift = np.zeros(n_param)       # Time steps since last partial drift detection
        self.alpha = None                                            # Adaptive threshold for global concept drift detection
        self.partial_alpha = np.asarray([None] * self.n_param)       # Adaptive threshold for partial concept drift detection
        self.mu_w = np.ones((self.M, self.n_param)) * init_mu        # Parameter Mean in window
        self.sigma_w = np.ones((self.M, self.n_param)) * init_sigma  # Parameter Variance in window
        self.param_sum = np.zeros((self.M - 1, self.n_param))        # Sum-expression for computation of moving average (see Eq. (8) in [1])
        self.global_info_ma = []                                     # Global moving average
        self.partial_info_ma = []                                    # Partial moving average

        # Parameters of FIRES (Probit) model according to [2]
        if self.base_model == 'probit':
            self.fires_mu = np.ones(self.n_param) * init_mu
            self.fires_sigma = np.ones(self.n_param) * init_sigma
            self.fires_epochs = epochs
            self.fires_lr_mu = lr_mu
            self.fires_lr_sigma = lr_sigma
            self.fires_labels = []                                          # Unique labels (fires requires binary labels)
            self.fires_encode_labels = True                                 # Indicator for warning message (auto-encoded labels)

        self.global_drift_detected = False
        self.partial_drift_detected = False
        self.partial_drift_features = None

    def reset(self):
        """
        ERICS need not be reset after a drift was detected
        """
        pass

    def partial_fit(self, X, y):
        """
        Process data batch and check for concept drift.

        Args:
            X (np.ndarray): Batch of observations
            y (np.ndarray): Batch of labels

        Returns:
            (bool, bool, float): indicator global drift, indicator partial drift, computation time in sec.
        """
        self.global_drift_detected = False
        self.global_drift_detected = False

        # Update alpha (Eq. 7 in [1])
        if self.alpha is not None:
            self.alpha -= (self.alpha * self.beta * self.time_since_last_global_drift)
        for k in range(self.n_param):  # partial alpha
            if self.partial_alpha[k] is not None:
                self.partial_alpha[k] -= (self.partial_alpha[k] * self.beta * self.time_since_last_partial_drift[k])

        # Update time since drift
        self.time_since_last_global_drift += 1
        self.time_since_last_partial_drift += 1

        # Update Parameter distribution
        if self.base_model == 'probit':
            self.__update_probit(X, y)  # Probit model
        else:
            raise NotImplementedError('The base model {} has not been implemented.'.format(self.base_model))

        self.__update_param_sum()                   # Update the sum expression for observations in a shifting window
        self.__compute_moving_average()             # Compute moving average in specified window
        self.global_drift_detected, self.partial_drift_detected, self.partial_drift_features = self.__detect_drift()    # Detect concept drift

        # Update time step
        self.time_step += 1

    def detected_global_change(self):
        return self.global_drift_detected

    def detected_partial_change(self):
        """
        Check if partial change was detected.

        Returns:
             bool: indicator whether partial change was detected
             list: indices of input features with detected drift
        """
        return self.partial_drift_detected, self.partial_drift_features

    def detected_warning_zone(self):
        pass

    def get_length_estimation(self):
        pass

    # ----------------------------------------
    # ERICS Functionality (left unchanged)
    # ----------------------------------------
    def __update_param_sum(self):
        """
        Retrieve current parameter distribution and compute sum expression according to Eq. (8) in the ERICS paper [1]
        """
        # Retrieve current distribution parameters
        if self.base_model == 'probit':
            new_mu = copy.copy(self.fires_mu).reshape(1, -1)
            new_sigma = copy.copy(self.fires_sigma).reshape(1, -1)
        else:
            raise NotImplementedError('The base model {} has not been implemented.'.format(self.base_model))

        # Drop oldest entry from window
        self.mu_w = self.mu_w[1:, :]
        self.sigma_w = self.sigma_w[1:, :]

        # Add new entry to window
        self.mu_w = np.concatenate((self.mu_w, new_mu))
        self.sigma_w = np.concatenate((self.sigma_w, new_sigma))

        # Compute parameter sum expression
        for t in range(self.M - 1):
            self.param_sum[t, :] = (self.sigma_w[t + 1, :] ** 2 + (self.mu_w[t, :] - self.mu_w[t + 1, :]) ** 2) / self.sigma_w[t, :] ** 2

    def __compute_moving_average(self):
        """
        Compute the moving average (according to Eq. (8) in the ERICS paper [1])
        """
        partial_ma = np.zeros(self.n_param)
        global_score = np.zeros(self.M - 1)

        for k in range(self.n_param):
            partial_score = self.param_sum[:, k] - 1
            global_score += partial_score
            partial_ma[k] = np.sum(np.abs(partial_score)) / (2 * self.M)  # Add partial mov. avg. for parameter k

        global_ma = np.sum(np.abs(global_score)) / (2 * self.M)

        self.global_info_ma.append(global_ma)
        self.partial_info_ma.append(partial_ma)

    def __detect_drift(self):
        """
        Detect global and partial concept drift using the adaptive alpha-threshold
        :return: global drift indicator, partial drift indicator, list of features with detected drift
        :rtype: bool, bool, list
        """
        global_window_delta = None
        partial_window_delta = None

        # Compute delta in sliding window W (according to Eq. (5) in the ERICS paper [1])
        if self.W < 2:
            self.W = 2
            warn('Sliding window for concept drift detection was automatically set to 2 observations.')

        if len(self.global_info_ma) < self.W:
            oldest_entry = len(self.global_info_ma)
        else:
            oldest_entry = self.W

        if oldest_entry == 1:  # In case of only one observation
            global_window_delta = copy.copy(self.global_info_ma[-1])
            partial_window_delta = copy.copy(self.partial_info_ma[-1])
        else:
            for t in range(oldest_entry, 1, -1):
                if t == oldest_entry:
                    global_window_delta = self.global_info_ma[-t+1] - self.global_info_ma[-t]  # newer - older
                    partial_window_delta = self.partial_info_ma[-t+1] - self.partial_info_ma[-t]
                else:
                    global_window_delta += (self.global_info_ma[-t+1] - self.global_info_ma[-t])
                    partial_window_delta += (self.partial_info_ma[-t+1] - self.partial_info_ma[-t])

        # (Re-) Initialize alpha if it is None (at time step 0 or if a drift was detected)
        if self.alpha is None:
            self.alpha = np.abs(global_window_delta)  # according to Eq. (6) in [1] -> abs() is only required at t=0, to make sure that alpha > 0
        if None in self.partial_alpha:
            unspecified = np.isnan(self.partial_alpha.astype(float)).flatten()
            self.partial_alpha[unspecified] = np.abs(partial_window_delta[unspecified])

        # Global Drift Detection
        g_drift = False
        if global_window_delta > self.alpha:
            g_drift = True
            self.time_since_last_global_drift = 0
            self.alpha = None

        # Partial Drift Detection
        p_drift = False
        partial_drift_bool = partial_window_delta > self.partial_alpha
        partial_drift_features = np.argwhere(partial_drift_bool).tolist()
        for k in partial_drift_features:
            p_drift = True
            self.time_since_last_partial_drift[k] = 0
            self.partial_alpha[k] = None

        return g_drift, p_drift, partial_drift_features

    # ----------------------------------------
    # BASE MODELS
    # ----------------------------------------
    def __update_probit(self, x, y):
        """
        Update parameters of the Probit model
        According to [2], as implemented here https://github.com/haugjo/fires
        We have slightly adjusted the original code to fit our use case.
        :param x: (np.ndarray) Batch of observations (numeric values only, consider normalizing data for better results)
        :param y: (np.ndarray) Batch of labels: type binary, i.e. {-1,1} (bool, int or str will be encoded accordingly)
        """
        # Encode labels
        for y_val in np.unique(y):  # Add newly observed unique labels
            if y_val not in set(self.fires_labels):
                self.fires_labels.append(y_val)

        if tuple(self.fires_labels) != (-1, 1):  # Check if labels are encoded correctly
            if self.fires_encode_labels:
                warn('FIRES WARNING: The target variable will automatically be encoded as {-1, 1}.')
                self.fires_encode_labels = False  # set indicator to false

            if len(self.fires_labels) < 2:
                y[y == self.fires_labels[0]] = -1
            elif len(self.fires_labels) == 2:
                y[y == self.fires_labels[0]] = -1
                y[y == self.fires_labels[1]] = 1
            else:
                raise ValueError('The target variable y must be binary.')

        for epoch in range(self.fires_epochs):
            # Shuffle the observations
            random_idx = np.random.permutation(len(y))
            x = x[random_idx]
            y = y[random_idx]

            # Iterative update of mu and sigma
            try:
                # Helper functions
                dot_mu_x = np.dot(x, self.fires_mu)
                rho = np.sqrt(1 + np.dot(x ** 2, self.fires_sigma ** 2))

                # Gradients
                nabla_mu = norm.pdf(y / rho * dot_mu_x) * (y / rho * x.T)
                nabla_sigma = norm.pdf(y / rho * dot_mu_x) * (
                            - y / (2 * rho ** 3) * 2 * (x ** 2 * self.fires_sigma).T * dot_mu_x)

                # Marginal Likelihood
                marginal = norm.cdf(y / rho * dot_mu_x)

                # Update parameters
                self.fires_mu += self.fires_lr_mu * np.mean(nabla_mu / marginal, axis=1)
                self.fires_sigma += self.fires_lr_sigma * np.mean(nabla_sigma / marginal, axis=1)
            except TypeError as e:
                raise TypeError('All features must be a numeric data type.') from e
