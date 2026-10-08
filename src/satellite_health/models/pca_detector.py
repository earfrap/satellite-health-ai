
"""PCA-based multivariate anomaly detection for E2."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class RobustChannelScaler:
    """Scale telemetry channels using training medians and MADs."""

    def __init__(self):
        self.median_ = None
        self.scale_ = None
        self.columns_ = None

    def fit(self, X: pd.DataFrame):
        """Estimate robust scaling parameters from nominal training data."""
        if X.empty:
            raise ValueError("Training data cannot be empty.")

        if X.isna().any().any():
            raise ValueError("Training data contain missing values.")

        if not np.isfinite(X.to_numpy()).all():
            raise ValueError("Training data contain non-finite values.")

        median = X.median()
        mad = (X - median).abs().median()

        if (mad <= 0).any():
            raise ValueError("At least one channel has zero MAD.")

        self.median_ = median
        self.scale_ = 1.4826 * mad
        self.columns_ = list(X.columns)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply training-derived scaling to new observations."""
        if self.median_ is None:
            raise RuntimeError("Scaler must be fitted first.")

        if list(X.columns) != self.columns_:
            raise ValueError("Input channels differ from training.")

        if X.isna().any().any():
            raise ValueError("Input data contain missing values.")

        if not np.isfinite(X.to_numpy()).all():
            raise ValueError("Input data contain non-finite values.")

        return (X - self.median_) / self.scale_




class PCAAnomalyDetector:
    """Multivariate PCA detector using squared prediction error."""

    def __init__(self, n_components: int = 2):
        if not isinstance(n_components, int) or isinstance(n_components, bool):
            raise ValueError("n_components must be an integer.")

        if n_components < 1:
            raise ValueError("n_components must be positive.")

        self.n_components = n_components
        self.scaler_ = None
        self.pca_ = None

    def fit(self, X: pd.DataFrame):
        """Fit robust scaling and PCA on nominal training data."""
        if self.n_components >= X.shape[1]:
            raise ValueError(
                "SPE requires fewer components than input channels."
            )

        scaler = RobustChannelScaler().fit(X)
        Z = scaler.transform(X)

        pca = PCA(n_components=self.n_components)
        pca.fit(Z)

        self.scaler_ = scaler
        self.pca_ = pca

        return self

    def score_samples(self, X: pd.DataFrame) -> pd.Series:
        """Return squared reconstruction error for each timestamp."""
        if self.pca_ is None:
            raise RuntimeError("Detector must be fitted first.")

        Z = self.scaler_.transform(X)

        projected = self.pca_.transform(Z)
        reconstructed = self.pca_.inverse_transform(projected)

        residual = Z.to_numpy() - reconstructed
        spe = np.sum(residual ** 2, axis=1)

        return pd.Series(
            spe,
            index=X.index,
            name="spe_score",
        )

