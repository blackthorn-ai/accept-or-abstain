"""Descriptive gate reports; repeated query-seed rows are not independent trials."""
from __future__ import annotations

import numpy as np
import pandas as pd


def gate_reports(long: pd.DataFrame, held_out: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-seed counts and descriptive pooled summaries for Gate B.

    Pooling weights each accepted prediction-seed observation equally. It does
    not estimate precision across independent molecules, chemical groups or future
    campaigns, and no independent-binomial confidence interval is attached.
    """
    keys = ["task", "model", "seed", "cal", "test"]
    tests = pd.DataFrame(held_out, columns=["family", "task", "test"])
    base = long[long.gate == "none"][keys + ["n"]].rename(columns={"n": "n_test"})
    d = long[long.gate == "LTT_fused"].merge(tests, on=["task", "test"], validate="many_to_one")
    d = d.merge(base, on=keys, how="left", validate="many_to_one")
    if d.empty or d.n_test.isna().any():
        raise ValueError("Gate rows need corresponding nonempty ungated test counts")
    if d.duplicated(keys + ["target"]).any():
        raise ValueError("Duplicate seed/calibration/target gate rows")
    counts = d[["n_test", "n_acc"]].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Accepted/test counts must be finite integers")
    if ((d.n_test <= 0) | (d.n_acc < 0) | (d.n_acc > d.n_test)).any():
        raise ValueError("Invalid accepted/test counts")
    if not np.allclose(d.coverage, d.n_acc / d.n_test, rtol=0, atol=1e-12):
        raise ValueError("Coverage disagrees with accepted/test counts")
    active = d.n_acc > 0
    if d.loc[active, "precision"].isna().any() or d.loc[~active, "precision"].notna().any():
        raise ValueError("Precision must be undefined exactly when the gate abstains")
    correct = d.precision.fillna(0) * d.n_acc
    if not np.allclose(correct, np.rint(correct), rtol=0, atol=1e-8):
        raise ValueError("Precision does not recover an integer correct count")
    d["n_correct"] = np.rint(correct).astype(int)
    if ((d.n_correct < 0) | (d.n_correct > d.n_acc)).any():
        raise ValueError("Invalid correct counts")
    d[["n_acc", "n_test"]] = d[["n_acc", "n_test"]].astype(int)
    cols = ["family", *keys, "target", "n_test", "n_acc", "n_correct", "coverage", "precision"]
    per_seed = d[cols].sort_values(["model", "target", "family", "task", "test", "seed", "cal"])
    d["accepting"] = active.astype(int)
    cells = ["family", "task", "test", "model", "cal", "target"]
    pooled = d.groupby(cells, sort=True).agg(
        n_prediction_seed=("n_test", "sum"),
        n_acc=("n_acc", "sum"), n_correct=("n_correct", "sum"),
        coverage=("coverage", "mean"), seeds=("seed", "count"),
        seeds_accepting=("accepting", "sum"), mean_precision=("precision", "mean"),
    ).reset_index()
    pooled["pooled_precision"] = pooled.n_correct / pooled.n_acc.replace(0, np.nan)
    return per_seed.reset_index(drop=True), pooled
