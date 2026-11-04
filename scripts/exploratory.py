"""Exploratory analyses (not in PROTOCOL.md; computed after the hypotheses were scored).

Writes results/exploratory.json and results/exploratory_conformal.csv:
  - pooled precision of every gate cell (all accepted queries across seeds)
  - per-seed target attainment of Gate B, by calibration fold
  - conformal intervals calibrated on C_h: coverage and width relative to C_r
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyse import HELD_OUT, LABEL, tol
from aoa import gates

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PRED = RES / "predictions"


def main(model="RF"):
    M = pd.read_csv(RES / "metrics_long.csv")
    M = M[M.model == model]
    out = {"model": model}
    # per-seed target attainment of Gate B on the held-out tests
    att = {}
    for cal in ("C_r", "C_h"):
        n = met = 0
        for _, task, test in HELD_OUT:
            d = M[(M.task == task) & (M.test == test) & (M.cal == cal) & (M.gate == "LTT_fused") & (M.n_acc > 0)]
            n += len(d)
            met += int((d.precision >= d.target).sum())
        att[cal] = {"seed_cells_accepting": n, "met_target": met}
    out["per_seed_attainment"] = att
    # conformal: C_h-calibrated vs C_r-calibrated intervals on held-out tests
    rows = []
    for _, task, test in HELD_OUT + [("", "logp", "R")]:
        cov = {"C_r": [], "C_h": []}
        wid = {"C_r": [], "C_h": []}
        for f in sorted(PRED.glob(f"{task}__{model}__s*.csv.gz")):
            df = pd.read_csv(f)
            T = df[df.set == test]
            for cal in ("C_r", "C_h"):
                C = df[df.set == cal]
                q, eps = gates.conformal_quantile(C.y, C.yhat, C.u, 0.1)
                cov[cal].append(gates.conformal_covered(T.y, T.yhat, T.u, q, eps).mean())
                wid[cal].append(float(np.mean(2 * q * (T.u + eps))))
        rows.append({"test": LABEL.get((task, test), f"{task}/{test}"),
                     "cov_C_r": np.mean(cov["C_r"]), "cov_C_h": np.mean(cov["C_h"]),
                     "width_C_r": np.mean(wid["C_r"]), "width_C_h": np.mean(wid["C_h"]),
                     "width_ratio": np.mean(wid["C_h"]) / np.mean(wid["C_r"])})
    C = pd.DataFrame(rows)
    C.to_csv(RES / f"exploratory_conformal_{model}.csv", index=False)
    out["conformal"] = C.round(3).to_dict(orient="records")
    (RES / f"exploratory_{model}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(att, indent=1))
    print(C.round(3).to_string(index=False))


if __name__ == "__main__":
    main("RF")
