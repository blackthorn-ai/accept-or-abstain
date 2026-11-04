"""Accept-or-abstain gates, Learn-then-Test thresholds and conformal intervals.

Protocol sections 6-7. A *score* is any per-query number where larger means "more
likely to be correct": s1 (similarity), -u (negative spread), or the fused
probability of Gate B.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression

N_MIN = 50  # strictest candidate threshold must accept at least this many points


def fused_features(s1, s5, u):
    return np.column_stack([s1, s5, np.log(np.asarray(u) + 1e-6)])


@dataclass
class FusedScorer:
    """Gate B score: logistic model of P(correct | s1, s5, log u)."""

    model: LogisticRegression

    @classmethod
    def fit(cls, s1, s5, u, correct):
        y = np.asarray(correct).astype(int)
        if y.min() == y.max():  # degenerate fold: a constant model
            m = LogisticRegression()
            m.classes_ = np.array([0, 1])
            m.coef_ = np.zeros((1, 3))
            m.intercept_ = np.array([np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3))])
            return cls(m)
        m = LogisticRegression(C=1.0, max_iter=1000).fit(fused_features(s1, s5, u), y)
        return cls(m)

    def score(self, s1, s5, u):
        return self.model.predict_proba(fused_features(s1, s5, u))[:, 1]


def ltt_threshold(score_cal, correct_cal, target: float, delta: float = 0.1) -> float:
    """Learn-then-Test threshold by fixed-sequence binomial testing.

    Candidates are the distinct calibration scores, from strict (high) to loose. The
    sequence starts at the strictest threshold accepting at least N_MIN points (or all
    points, if fewer exist). Returns +inf if no threshold passes (abstain on all).
    """
    s = np.asarray(score_cal, dtype=float)
    c = np.asarray(correct_cal).astype(int)
    order = np.argsort(-s, kind="stable")
    s_sorted, c_sorted = s[order], c[order]
    k_cum = np.cumsum(c_sorted)
    n_total = len(s)
    # candidate cut positions: accept the top-n points, at the last index of each tie block
    distinct_end = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), n_total - 1]
    best = np.inf
    start_n = min(N_MIN, n_total)
    for idx in distinct_end:
        n = idx + 1
        if n < start_n:
            continue
        k = k_cum[idx]
        p = binom.sf(k - 1, n, target)  # P(Bin(n, target) >= k)
        if p <= delta:
            best = s_sorted[idx]
        else:
            break
    return best


def empirical_threshold(score_cal, correct_cal, target: float) -> float:
    """Loosest threshold whose empirical calibration precision is >= target (no guarantee)."""
    s = np.asarray(score_cal, dtype=float)
    c = np.asarray(correct_cal).astype(int)
    order = np.argsort(-s, kind="stable")
    prec = np.cumsum(c[order]) / np.arange(1, len(s) + 1)
    ok = np.flatnonzero(prec >= target)
    return np.inf if len(ok) == 0 else s[order][ok.max()]


def conjunctive_gate(s1_cal, u_cal, correct_cal, target: float, grid: int = 20):
    """Gate A: accept if s1 >= t1 and u <= t2, maximising coverage at empirical precision >= t."""
    qs = np.linspace(0, 1, grid)
    t1s = np.quantile(s1_cal, qs)
    t2s = np.quantile(u_cal, qs)
    c = np.asarray(correct_cal).astype(bool)
    best = (np.inf, -np.inf, 0.0)
    for t1 in t1s:
        for t2 in t2s:
            acc = (s1_cal >= t1) & (u_cal <= t2)
            n = acc.sum()
            if n == 0:
                continue
            if c[acc].mean() >= target and n / len(c) > best[2]:
                best = (t1, t2, n / len(c))
    return best[0], best[1]


def conformal_quantile(y_cal, yhat_cal, u_cal, alpha: float = 0.1):
    """Normalised split-conformal quantile and epsilon (protocol section 6)."""
    eps = 0.1 * float(np.median(u_cal))
    scores = np.abs(np.asarray(y_cal) - np.asarray(yhat_cal)) / (np.asarray(u_cal) + eps)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher")), eps


def conformal_covered(y, yhat, u, q, eps):
    return np.abs(np.asarray(y) - np.asarray(yhat)) <= q * (np.asarray(u) + eps)
