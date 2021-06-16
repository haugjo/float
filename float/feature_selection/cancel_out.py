from float.feature_selection.feature_selector import FeatureSelector
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class CancelOutFeatureSelector(FeatureSelector):
    def __init__(self, n_total_features, n_selected_features):
        super().__init__(n_total_features, n_selected_features, supports_multi_class=False, supports_streaming_features=False)

    def weight_features(self, X, y):
        """
        Given a batch of observations and corresponding labels, computes feature weights.

        Args:
            X (np.ndarray): samples of current batch
            y (np.ndarray): labels of current batch
        """
        self.raw_weight_vector = self._train_ann(X, y, 50)

    @staticmethod
    def _train_ann(X, y, num_epochs):
        model = NeuralNet(X.shape[1], X.shape[1] + 1, 2)

        n_features = X.shape[1]
        mydataset = MyDataset(X, y)
        batch_size = int(n_features / 5)

        train_loader = torch.utils.data.DataLoader(dataset=mydataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam([
            {"params": model.CancelOut.parameters(), "lr": 0.01},
            {"params": model.fc1.parameters(), "lr": 0.003},
            {"params": model.fc2.parameters(), "lr": 0.003},
        ])
        patience = 3

        early_stopping = EarlyStopping(patience=patience, verbose=False)

        avg_train_losses = []
        train_losses = []

        for epoch in range(num_epochs):
            for i, (sample, labels) in enumerate(train_loader):
                # Forward pass
                outputs = model(sample.float())

                weights_co = list(model.CancelOut.parameters())[0]
                reg = torch.var(weights_co)
                nrm = torch.norm(weights_co, 1)  # torch.sum(torch.abs(weights_co))

                loss = criterion(outputs, labels.long()) - 0.001 * (reg / n_features) + 0.0001 * (nrm / n_features)
                train_losses.append(loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # data for early stoping
            train_loss = np.average(train_losses)
            avg_train_losses.append(train_loss)
            early_stopping(train_loss, model)
            if early_stopping.early_stop:
                # print("Early stopping")
                break

        return list(model.CancelOut.parameters())[0].detach().numpy()

    def detect_concept_drift(self, x, y):
        raise NotImplementedError


class NeuralNet(nn.Module):
    """ANN"""

    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initializes the neural network.

        Args:
            input_size (int): size of the input layer
            hidden_size (int): hidden size of ANN
            num_classes (int): output size of ANN
        """

        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU6()
        self.CancelOut = CancelOut(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x1 = self.CancelOut(x)
        x2 = self.fc1(x1)
        x3 = self.relu(x2)
        x4 = self.fc2(x3)
        return x4


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    # https: // github.com / Bjarten / early - stopping - pytorch

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            # if self.verbose:
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class CancelOut(nn.Module):
    """
    CancelOut layer.
    """
    def __init__(self, input_size, *args, **kwargs):
        """
        Initializes the Cancel Out layer.

        Args:
            input_size (int): size of the input layer
        """
        super(CancelOut, self).__init__()
        self.weights = nn.Parameter(torch.zeros(input_size, requires_grad=True))

    def forward(self, X):
        """
        Performs a forward pass.

        Args:
            X (np.ndarray): samples of the current batch

        Returns:
            np.ndarray: X * sigmoid(weights)
        """
        return X * torch.sigmoid(self.weights.float() + 2)


class MyDataset(Dataset):
    """
    Dataset Class for PyTorch model.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)
