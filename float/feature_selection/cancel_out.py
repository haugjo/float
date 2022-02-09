"""CancelOut Feature Selection Method.

This module contains an adaptation of the CancelOut feature selection method for data streams. CancelOut
was introduced as a feature selection layer for neural networks by:
BORISOV, Vadim; HAUG, Johannes; KASNECI, Gjergji. Cancelout: A layer for feature selection in deep neural networks.
In: International Conference on Artificial Neural Networks. Springer, Cham, 2019. S. 72-83.

Copyright (C) 2022 Johannes Haug.
"""
import numpy as np
from numpy.typing import ArrayLike
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Union

from float.feature_selection.base_feature_selector import BaseFeatureSelector


class CancelOutFeatureSelector(BaseFeatureSelector):
    """CancelOut feature selector."""
    def __init__(self,
                 n_total_features: int,
                 n_selected_features: int,
                 reset_after_drift: bool = False,
                 baseline: str = 'constant',
                 ref_sample: Union[float, ArrayLike] = 0):
        """Inits the feature selector.

        Args:
            n_total_features: The total number of features.
            n_selected_features: The number of selected features.
            reset_after_drift: A boolean indicating if the change detector will be reset after a drift was detected.
            baseline:
                A string identifier of the baseline method. The baseline is the value that we substitute non-selected
                features with. This is necessary, because most online learning models are not able to handle arbitrary
                patterns of missing data.
            ref_sample:
                A sample used to compute the baseline. If the constant baseline is used, one needs to provide a single
                float value.
        """
        super().__init__(n_total_features=n_total_features,
                         n_selected_features=n_selected_features,
                         supports_multi_class=False,
                         reset_after_drift=reset_after_drift,
                         baseline=baseline,
                         ref_sample=ref_sample)

    def weight_features(self, X: ArrayLike, y: ArrayLike):
        """Updates feature weights.

        Args:
            X: Array/matrix of observations.
            y: Array of corresponding labels.
        """
        # We train the ANN with default parameters proposed by the authors
        self.weights = self._train_ann(X=X, y=y, num_epochs=50, batch_size=128)

    def reset(self):
        """Resets the feature selector.

        CancelOut does not need to be reset, since the DNN is trained anew at every training iteration.
        """
        pass

    # ----------------------------------------
    # CancelOut Functionality
    # ----------------------------------------
    @staticmethod
    def _train_ann(X: ArrayLike, y: ArrayLike, num_epochs: int, batch_size: int) -> ArrayLike:
        """Trains a neural network and returns feature weights from the CancelOut layer.

        Args:
            X: Array/matrix of observations.
            y: Array of corresponding labels.
            num_epochs: Number of training epochs.
            batch_size: The training batch size.

        Returns:
            ArrayLike: A vector of feature weights
        """
        model = CancelOutNeuralNet(X.shape[1], X.shape[1] + 10, 2)
        data_loader = CancelOutDataLoader(X, y)
        batch_size = batch_size
        train_loader = torch.utils.data.DataLoader(dataset=data_loader,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(num_epochs):
            for i, (sample, labels) in enumerate(train_loader):
                # Forward pass
                outputs = model(sample.float())

                # cancelout regularization
                weights_co = torch.sigmoid(list(model.cancelout.parameters())[0])
                l1_norm = torch.norm(weights_co, 1)
                l2_norm = torch.norm(weights_co, 2)
                lambda_1 = 0.001
                lambda_2 = 0.001
                loss = criterion(outputs, labels) + lambda_1 * l1_norm + lambda_2 * l2_norm
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

        return list(model.cancelout.parameters())[0].detach().numpy() / np.sum(
            list(model.cancelout.parameters())[0].detach().numpy())


# ----------------------------------------
# CancelOut Classes
# ----------------------------------------
class CancelOutNeuralNet(nn.Module):
    """A neural network with CancelOut layer."""
    def __init__(self, input_size, hidden_size, num_classes):
        super(CancelOutNeuralNet, self).__init__()
        self.cancelout = CancelOut(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.cancelout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class CancelOutDataLoader(Dataset):
    """CancelOut dataset loader for the neural network."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class CancelOut(nn.Module):
    """CancelOut network layer."""
    def __init__(self, x):
        """Initializes the network layer

        Args:
            X: an input data (vector, matrix, tensor)
        """
        super(CancelOut, self).__init__()
        self.weights = nn.Parameter(torch.zeros(x, requires_grad=True) + 4)

    def forward(self, x):
        return x * torch.sigmoid(self.weights.float())
