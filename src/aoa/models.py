"""Base models with an ensemble spread (protocol section 4)."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class RFEnsemble:
    """M1: random forest; spread is the standard deviation of per-tree predictions."""

    name = "RF"

    def __init__(self, seed: int):
        self.model = RandomForestRegressor(
            n_estimators=300, max_features=0.2, n_jobs=-1, random_state=seed
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        per_tree = np.stack([t.predict(X) for t in self.model.estimators_])
        return per_tree.mean(0), per_tree.std(0)


class MLPEnsemble:
    """M2: five MLPs (512-128); spread is the standard deviation across members."""

    name = "MLP"

    def __init__(self, seed: int, n_members: int = 5):
        self.members = [
            MLPRegressor(
                hidden_layer_sizes=(512, 128),
                alpha=1e-4,
                early_stopping=True,
                validation_fraction=0.1,
                max_iter=200,
                random_state=seed * 100 + i,
            )
            for i in range(n_members)
        ]
        self.xs = StandardScaler()
        self.y_mu = 0.0
        self.y_sd = 1.0

    def fit(self, X, y):
        Xs = self.xs.fit_transform(X)
        self.y_mu, self.y_sd = float(np.mean(y)), float(np.std(y) or 1.0)
        ys = (y - self.y_mu) / self.y_sd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for m in self.members:
                m.fit(Xs, ys)
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        Xs = self.xs.transform(X)
        P = np.stack([m.predict(Xs) for m in self.members]) * self.y_sd + self.y_mu
        return P.mean(0), P.std(0)


MODELS = {"RF": RFEnsemble, "MLP": MLPEnsemble}
