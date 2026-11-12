"""Exploratory (not in the protocol): spread growth and C_r-calibrated interval width, by learner.

Written after M3 was scored under amendment 2.0, to explain its H4 table. For each held-out
test and learner: the ratio of mean spread on the held-out test to mean spread on the random
test R; the coverage of the C_r-calibrated intervals on the held-out test; and their mean
width. Writes results/amendment2/exploratory_spread_and_intervals.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analyse import HELD_OUT, LABEL
from aoa import gates

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (("RF", "results/predictions"), ("MLP", "results/predictions"), ("DMPNN", "results/predictions_m3"))


def main():
    rows = []
    for model, pdir in SOURCES:
        for fam, task, test in HELD_OUT:
            ur, uh, wh, ch = [], [], [], []
            for f in sorted((ROOT / pdir).glob(f"{task}__{model}__s*.csv.gz")):
                d = pd.read_csv(f)
                C, R, T = d[d.set == "C_r"], d[d.set == "R"], d[d.set == test]
                q, eps = gates.conformal_quantile(C.y, C.yhat, C.u, 0.1)
                ur.append(R.u.mean())
                uh.append(T.u.mean())
                wh.append(np.mean(2 * q * (T.u + eps)))
                ch.append(gates.conformal_covered(T.y, T.yhat, T.u, q, eps).mean())
            rows.append({"model": model, "family": fam, "test": LABEL[(task, test)],
                         "spread_ratio_H_to_R": np.mean(uh) / np.mean(ur),
                         "coverage_H": np.mean(ch), "mean_width_H": np.mean(wh)})
    out = pd.DataFrame(rows)
    dest = ROOT / "results" / "amendment2" / "exploratory_spread_and_intervals.csv"
    out.to_csv(dest, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
