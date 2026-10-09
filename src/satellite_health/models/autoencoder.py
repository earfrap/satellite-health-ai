"""Small feed-forward autoencoder for spacecraft telemetry."""

import numpy as np
import pandas as pd
import torch
from torch import nn

from satellite_health.models.pca_detector import RobustChannelScaler


class AutoencoderNetwork(nn.Module):
    """Six-channel autoencoder with a two-dimensional bottleneck."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 6),
        )

    def forward(self, x):
        return self.network(x)


class AutoencoderDetector:
    """Reconstruction-error detector with robust input scaling."""

    def __init__(self, epochs=20, batch_size=1024, learning_rate=0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scaler_ = None
        self.model_ = None
        self.loss_history_ = []

    def fit(self, X: pd.DataFrame):
        torch.manual_seed(42)
        torch.set_num_threads(1)

        scaler = RobustChannelScaler().fit(X)
        Z = scaler.transform(X).to_numpy(dtype=np.float32).copy()
        data = torch.from_numpy(Z)

        model = AutoencoderNetwork()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate
        )
        loss_function = nn.MSELoss()

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0

            for batch in data.split(self.batch_size):
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = loss_function(reconstructed, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)

            mean_loss = total_loss / len(data)
            self.loss_history_.append(mean_loss)

        model.eval()
        self.scaler_ = scaler
        self.model_ = model
        return self

    def score_samples(self, X: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("Detector must be fitted first.")

        Z = self.scaler_.transform(X).to_numpy(dtype=np.float32).copy()
        data = torch.from_numpy(Z)

        errors = []

        with torch.no_grad():
            for batch in data.split(self.batch_size):
                reconstructed = self.model_(batch)
                error = ((batch - reconstructed) ** 2).sum(dim=1)
                errors.append(error.numpy())

        return pd.Series(
            np.concatenate(errors),
            index=X.index,
            name="autoencoder_score",
        )
