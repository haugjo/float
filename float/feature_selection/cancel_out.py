from float.feature_selection.base_feature_selector import BaseFeatureSelector
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class CancelOutFeatureSelector(BaseFeatureSelector):
    def __init__(self, n_total_features, n_selected_features, evaluation_metrics=None):
        """
        Initializes the Cancel Out feature selector.

        Args:
            n_total_features (int): total number of features
            n_selected_features (int): number of selected features
            evaluation_metrics (dict of str: function | dict of str: (function, dict)): {metric_name: metric_function} OR {metric_name: (metric_function, {param_name1: param_val1, ...})} a dictionary of metrics to be used
        """
        super().__init__(n_total_features, n_selected_features, evaluation_metrics, supports_multi_class=False,
                         supports_streaming_features=False)

    def weight_features(self, X, y):
        self.raw_weight_vector = self.__train_ann(X, y, 50, 128)

    @staticmethod
    def __train_ann(X, y, num_epochs, batch_size):
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


class CancelOutNeuralNet(nn.Module):
    """
    A simple DL model.
    """
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
    """
    A dataset loader.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class CancelOut(nn.Module):
    """
    A CancelOut Layer.
    """
    def __init__(self, X, *args, **kwargs):
        """

        Args:
            X: an input data (vector, matrix, tensor)
        """
        super(CancelOut, self).__init__()
        self.weights = nn.Parameter(torch.zeros(X, requires_grad=True) + 4)

    def forward(self, X):
        return X * torch.sigmoid(self.weights.float())
