import numpy as np
from warnings import warn
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from float.feature_selection.feature_selector import FeatureSelector


class FIRES(FeatureSelector):
    def __init__(self, n_total_features, n_selected_features, target_values, mu_init=0, sigma_init=1, penalty_s=0.01, penalty_r=0.01, epochs=1,
                 lr_mu=0.01, lr_sigma=0.01, scale_weights=True, model='probit'):
        """
        FIRES: Fast, Interpretable and Robust Evaluation and Selection of features
        cite: Haug et al. 2020. Leveraging Model Inherent Variable Importance for Stable Online Feature Selection.
        In Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’20),
        August 23–27, 2020, Virtual Event, CA, USA.

        Args:
            n_total_features (int): total number of features
            n_selected_features (int): number of selected features
            target_values (np.ndarray): unique target values (class labels)
            mu_init (int/np.ndarray): initial importance parameter
            sigma_init (int/np.ndarray): initial uncertainty parameter
            penalty_s (float): penalty factor for the uncertainty (corresponds to gamma_s in the paper)
            penalty_r (float): penalty factor for the regularization (corresponds to gamma_r in the paper)
            epochs (int): number of epochs that we use each batch of observations to update the parameters
            lr_mu (float): learning rate for the gradient update of the importance
            lr_sigma (float): learning rate for the gradient update of the uncertainty
            scale_weights (bool): if True, scale feature weights into the range [0,1]
            model (str): name of the base model to compute the likelihood (default is 'probit')
        """
        super().__init__(n_total_features, n_selected_features, supports_multi_class=False, supports_streaming_features=False)
        self.n_total_ftr = n_total_features
        self.target_values = target_values
        self.mu = np.ones(n_total_features) * mu_init
        self.sigma = np.ones(n_total_features) * sigma_init
        self.penalty_s = penalty_s
        self.penalty_r = penalty_r
        self.epochs = epochs
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.scale_weights = scale_weights
        self.model = model

        # Additional model-specific parameters
        self.model_param = {}

        # Probit model
        if self.model == 'probit' and tuple(target_values) != (-1, 1):
            if len(np.unique(target_values)) == 2:
                self.model_param['probit'] = True  # Indicates that we need to encode the target variable into {-1,1}
                warn('FIRES WARNING: The target variable will be encoded as: {} = -1, {} = 1'.format(
                    self.target_values[0], self.target_values[1]))
            else:
                raise ValueError('The target variable y must be binary.')

    def weight_features(self, X, y):
        """
        Given a batch of observations and corresponding labels, computes feature weights.

        Args:
            X (np.ndarray): samples of current batch
            y (np.ndarray): labels of current batch

        Returns:
            np.ndarray: feature weights
        """
        # Update estimates of mu and sigma given the predictive model
        if self.model == 'probit':
            self.__probit(X, y)
        # ### ADD YOUR OWN MODEL HERE ##################################################
        # elif self.model == 'your_model':
        #    self.__yourModel(x, y)
        ################################################################################
        else:
            raise NotImplementedError('The given model name does not exist')

        # Limit sigma to range [0, inf]
        if sum(n < 0 for n in self.sigma) > 0:
            self.sigma[self.sigma < 0] = 0
            warn('Sigma has automatically been rescaled to [0, inf], because it contained negative values.')

        # Compute feature weights
        self.raw_weight_vector = self.__compute_weights()

    def __probit(self, X, y):
        """
        Updates the distribution parameters mu and sigma by optimizing them in terms of the (log) likelihood.
        Here we assume a Bernoulli distributed target variable. We use a Probit model as our base model.
        This corresponds to the FIRES-GLM model in the paper.

        Args:
            X (np.ndarray): batch of observations (numeric values only, consider normalizing data for better results)
            y (np.ndarray): batch of labels: type binary, i.e. {-1,1} (bool, int or str will be encoded accordingly)
        """

        for epoch in range(self.epochs):
            # Shuffle the observations
            random_idx = np.random.permutation(len(y))
            X = X[random_idx]
            y = y[random_idx]

            # Encode target as {-1,1}
            if 'probit' in self.model_param:
                y[y == self.target_values[0]] = -1
                y[y == self.target_values[1]] = 1

            # Iterative update of mu and sigma
            try:
                # Helper functions
                dot_mu_x = np.dot(X, self.mu)
                rho = np.sqrt(1 + np.dot(X ** 2, self.sigma ** 2))

                # Gradients
                nabla_mu = norm.pdf(y / rho * dot_mu_x) * (y / rho * X.T)
                nabla_sigma = norm.pdf(y / rho * dot_mu_x) * (
                        - y / (2 * rho ** 3) * 2 * (X ** 2 * self.sigma).T * dot_mu_x)

                # Marginal Likelihood
                marginal = norm.cdf(y / rho * dot_mu_x)

                # Update parameters
                self.mu += self.lr_mu * np.mean(nabla_mu / marginal, axis=1)
                self.sigma += self.lr_sigma * np.mean(nabla_sigma / marginal, axis=1)
            except TypeError as e:
                raise TypeError('All features must be a numeric data type.') from e

    def __compute_weights(self):
        """
        Computes optimal weights according to the objective function proposed in the paper.
        We compute feature weights in a trade-off between feature importance and uncertainty.
        Thereby, we aim to maximize both the discriminative power and the stability/robustness of feature weights.

        Returns:
            np.ndarray: feature weights
        """
        # Compute optimal weights
        weights = (self.mu ** 2 - self.penalty_s * self.sigma ** 2) / (2 * self.penalty_r)

        if self.scale_weights:  # Scale weights to [0,1]
            weights = MinMaxScaler().fit_transform(weights.reshape(-1, 1)).flatten()

        return weights
