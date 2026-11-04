"""Risk-coverage curves, AURC and regression metrics (protocol section 8)."""
from __future__ import annotations

import numpy as np


def regression(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    r = y - yhat
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {
        "rmse": float(np.sqrt((r**2).mean())),
        "mae": float(np.abs(r).mean()),
        "r2": float(1 - (r**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(y)),
    }


def risk_coverage(score, correct):
    """Selective error rate when accepting the top-k queries by score, k = 1..n."""
    order = np.argsort(-np.asarray(score, float), kind="stable")
    err = 1 - np.asarray(correct, float)[order]
    k = np.arange(1, len(err) + 1)
    return k / len(err), np.cumsum(err) / k


def aurc(score, correct) -> float:
    _, risk = risk_coverage(score, correct)
    return float(risk.mean())


def error_at_coverage(score, correct, coverage: float = 0.3) -> float:
    cov, risk = risk_coverage(score, correct)
    idx = max(0, int(np.ceil(coverage * len(cov))) - 1)
    return float(risk[idx])


def selective_rmse_curve(score, y, yhat):
    order = np.argsort(-np.asarray(score, float), kind="stable")
    se = ((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)[order]
    k = np.arange(1, len(se) + 1)
    return k / len(se), np.sqrt(np.cumsum(se) / k)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (float(c - h), float(c + h))
