"""Stage 2: gates, metrics and the pre-registered hypotheses (PROTOCOL.md 7-9).

Reads results/predictions/*.csv.gz and writes:
  results/metrics_long.csv     one row per task/model/seed/calibration/test/gate/target
  results/scores_long.csv      AURC and error-at-30% per score
  results/summary_*.csv        seed-averaged tables used in the paper
  results/hypotheses.md        H1-H4, pass or fail, with the numbers behind them
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aoa import gates, metrics

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "predictions"
OUT = ROOT / "results"
TARGETS = [0.8, 0.9]
DELTA = 0.1
N_BOOT = 1000
FULL: dict = {}
HELD_OUT = [
    ("D1", "logp", "H"), ("D1", "logp", "E1"), ("D1", "logp_noCF", "E2"),
    ("D3", "bh_Test1", "H"), ("D3", "bh_Test2", "H"), ("D3", "bh_Test3", "H"),
    ("D3", "bh_Test4", "H"), ("D3", "sm_ligand", "H"),
]
LABEL = {
    ("logp", "H"): "LogP cluster", ("logp", "E1"): "LogP prospective (E1)",
    ("logp_noCF", "E2"): "LogP fluorinated (E2)", ("bh_Test1", "H"): "BH Test1",
    ("bh_Test2", "H"): "BH Test2", ("bh_Test3", "H"): "BH Test3", ("bh_Test4", "H"): "BH Test4",
    ("sm_ligand", "H"): "SM ligand",
}


def tol(task: str) -> float:
    return 0.5 if task.startswith("logp") else 10.0


def test_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {s: g for s, g in df.groupby("set") if s not in ("C_r", "C_h")}
    if "E2" in out and "sp3_CF" in out["E2"]:
        e2 = out["E2"]
        out["E2_CF"] = e2[e2.sp3_CF.astype(bool)]
        out["E2_parent"] = e2[~e2.sp3_CF.astype(bool)]
    return out


def analyse_file(f: Path, keep_arrays: dict):
    task, model, s = f.name.replace(".csv.gz", "").split("__")
    seed = int(s[1:])
    df = pd.read_csv(f)
    df["correct"] = (df.y - df.yhat).abs() <= tol(task)
    rows, srows = [], []
    tests = test_sets(df)
    for cal_name in ("C_r", "C_h"):
        cal = df[df.set == cal_name].reset_index(drop=True)
        rng = np.random.default_rng(10_000 + seed)
        perm = rng.permutation(len(cal))
        A, Bh = cal.iloc[perm[: len(cal) // 2]], cal.iloc[perm[len(cal) // 2 :]]
        fused = gates.FusedScorer.fit(A.s1, A.s5, A.u, A.correct)
        score_fns = {
            "s1": lambda d: d.s1.values,
            "neg_u": lambda d: -d.u.values,
            "fused": lambda d: fused.score(d.s1, d.s5, d.u),
        }
        thr = {(k, t): gates.ltt_threshold(fn(Bh), Bh.correct, t, DELTA)
               for k, fn in score_fns.items() for t in TARGETS}
        conj = {t: gates.conjunctive_gate(cal.s1.values, cal.u.values, cal.correct.values, t) for t in TARGETS}
        q, eps = gates.conformal_quantile(cal.y, cal.yhat, cal.u, alpha=0.1)
        for tname, T in tests.items():
            base = dict(task=task, model=model, seed=seed, cal=cal_name, test=tname)
            reg = metrics.regression(T.y, T.yhat)
            rows.append({**base, "gate": "none", "target": np.nan, "coverage": 1.0,
                         "precision": T.correct.mean(), "n_acc": len(T), **reg})
            covered = gates.conformal_covered(T.y, T.yhat, T.u, q, eps)
            for k, fn in score_fns.items():
                sc = fn(T)
                srows.append({**base, "score": k, "aurc": metrics.aurc(sc, T.correct),
                              "err30": metrics.error_at_coverage(sc, T.correct, 0.3),
                              "err_all": 1 - T.correct.mean()})
                for t in TARGETS:
                    acc = sc >= thr[(k, t)]
                    n_acc = int(acc.sum())
                    rows.append({**base, "gate": f"LTT_{k}", "target": t, "coverage": acc.mean(),
                                 "precision": T.correct.values[acc].mean() if n_acc else np.nan,
                                 "n_acc": n_acc,
                                 "conf_cov_acc": covered[acc].mean() if n_acc else np.nan})
                if k == "fused" and cal_name == "C_h":
                    keep_arrays[(task, model, seed, tname)] = (sc, T.correct.values)
                FULL[(task, model, seed, cal_name, tname, k)] = {
                    "score": sc, "correct": T.correct.values, "y": T.y.values, "yhat": T.yhat.values,
                    "acc08": sc >= thr[(k, 0.8)], "acc09": sc >= thr[(k, 0.9)],
                    "covered": covered, **({"sp3_CF": T.sp3_CF.values.astype(bool)} if "sp3_CF" in T else {}),
                }
            for t in TARGETS:
                t1, t2 = conj[t]
                acc = (T.s1.values >= t1) & (T.u.values <= t2)
                n_acc = int(acc.sum())
                rows.append({**base, "gate": "A_conj", "target": t, "coverage": acc.mean(),
                             "precision": T.correct.values[acc].mean() if n_acc else np.nan,
                             "n_acc": n_acc})
            rows.append({**base, "gate": "conformal90", "target": 0.9, "coverage": np.nan,
                         "precision": np.nan, "n_acc": len(T), "conf_cov_all": covered.mean()})
    return rows, srows


def boot_h1(arrays: dict, task: str, model: str, test: str, rng) -> tuple[float, float, float]:
    """Seed-averaged err@30%(fused) - err(all), with a bootstrap 95% interval."""
    seeds = sorted(s for (tk, m, s, te) in arrays if tk == task and m == model and te == test)
    def diff(sc, c):
        return metrics.error_at_coverage(sc, c, 0.3) - (1 - c.mean())
    point = np.mean([diff(*arrays[(task, model, s, test)]) for s in seeds])
    bs = []
    for _ in range(N_BOOT):
        vals = []
        for s in seeds:
            sc, c = arrays[(task, model, s, test)]
            i = rng.integers(0, len(c), len(c))
            vals.append(diff(sc[i], c[i]))
        bs.append(np.mean(vals))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    files = sorted(PRED.glob("*.csv.gz"))
    arrays: dict = {}
    rows, srows = [], []
    for f in files:
        r, s = analyse_file(f, arrays)
        rows += r
        srows += s
    M = pd.DataFrame(rows)
    S = pd.DataFrame(srows)
    M.to_csv(OUT / "metrics_long.csv", index=False)
    import pickle
    (OUT / "cache").mkdir(exist_ok=True)
    with open(OUT / "cache" / "arrays.pkl", "wb") as fh:
        pickle.dump(FULL, fh)
    S.to_csv(OUT / "scores_long.csv", index=False)

    # seed-averaged summaries
    g = ["task", "model", "cal", "test", "gate", "target"]
    Msum = M.groupby(g, dropna=False).agg(
        coverage=("coverage", "mean"), precision=("precision", "mean"),
        precision_sd=("precision", "std"), n_seeds_defined=("precision", "count"),
        rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), mae=("mae", "mean"),
        r2=("r2", "mean"), r2_sd=("r2", "std"), n=("n", "mean"),
        conf_cov_all=("conf_cov_all", "mean"), conf_cov_acc=("conf_cov_acc", "mean"),
    ).reset_index()
    Msum.to_csv(OUT / "summary_gates.csv", index=False)
    Ssum = S.groupby(["task", "model", "cal", "test", "score"]).agg(
        aurc=("aurc", "mean"), aurc_sd=("aurc", "std"), err30=("err30", "mean"),
        err_all=("err_all", "mean")).reset_index()
    Ssum.to_csv(OUT / "summary_scores.csv", index=False)

    lines = ["# Pre-registered hypotheses: results", "",
             "Generated by `scripts/analyse.py` from the files in `results/predictions/`. "
             "Model M1 (RF) decides the hypotheses; M2 (MLP) is reported for robustness.", ""]
    verdicts = {}
    for model in ["RF", "MLP"]:
        if not (M.model == model).any():
            continue
        lines += [f"## Model {model}", ""]
        rng = np.random.default_rng(0)
        # H1
        lines += ["### H1: fused selective error at 30% coverage below the ungated error", "",
                  "| Held-out test | ungated error | error at 30% | difference [95% CI] | holds |",
                  "|---|---|---|---|---|"]
        h1_all = True
        for fam, task, test in HELD_OUT:
            key = [k for k in arrays if k[0] == task and k[1] == model and k[3] == test]
            if not key:
                h1_all = False
                lines.append(f"| {LABEL[(task, test)]} | missing | | | no |")
                continue
            d, lo, hi = boot_h1(arrays, task, model, test, rng)
            row = Ssum[(Ssum.task == task) & (Ssum.model == model) & (Ssum.cal == "C_h") &
                       (Ssum.test == test) & (Ssum.score == "fused")].iloc[0]
            ok = hi < 0
            h1_all &= ok
            lines.append(f"| {LABEL[(task, test)]} | {row.err_all:.3f} | {row.err30:.3f} | "
                         f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}] | {'yes' if ok else 'no'} |")
        lines += ["", f"**H1 {'holds' if h1_all else 'does not hold'}** (criterion: every held-out test).", ""]
        # H2
        lines += ["### H2: random-fold calibration misses the target; held-out-fold calibration meets it", "",
                  "| Held-out test | target | precision, C_r [coverage] | precision, C_h [coverage] |",
                  "|---|---|---|---|"]
        miss_r = meet_h = cells = 0
        for fam, task, test in HELD_OUT:
            for t in TARGETS:
                vals = {}
                for cal in ("C_r", "C_h"):
                    r = Msum[(Msum.task == task) & (Msum.model == model) & (Msum.cal == cal) &
                             (Msum.test == test) & (Msum.gate == "LTT_fused") & (Msum.target == t)]
                    vals[cal] = r.iloc[0] if len(r) else None
                if vals["C_r"] is None:
                    continue
                cells += 1
                pr, ph = vals["C_r"].precision, vals["C_h"].precision
                miss_r += (not np.isnan(pr)) and pr < t - 0.05
                meet_h += np.isnan(ph) or ph >= t - 0.05
                fmt = lambda v: "abstained" if np.isnan(v.precision) else f"{v.precision:.3f} [{v.coverage:.2f}]"
                lines.append(f"| {LABEL[(task, test)]} | {t} | {fmt(vals['C_r'])} | {fmt(vals['C_h'])} |")
        h2 = cells > 0 and miss_r >= cells / 2 and meet_h >= 0.75 * cells
        lines += ["", f"C_r misses by > 0.05 in {miss_r}/{cells} cells (needs ≥ {cells/2:g}); "
                  f"C_h meets t − 0.05 in {meet_h}/{cells} (needs ≥ {0.75*cells:g}). "
                  f"**H2 {'holds' if h2 else 'does not hold'}.**", ""]
        # H3
        lines += ["### H3: fused AURC at least 10% below the better single signal", "",
                  "| Held-out test | AURC s1 | AURC −u | AURC fused | relative reduction | ≥ 10% |",
                  "|---|---|---|---|---|---|"]
        wins = {"D1": 0, "D3": 0}
        for fam, task, test in HELD_OUT:
            sub = Ssum[(Ssum.task == task) & (Ssum.model == model) & (Ssum.cal == "C_h") & (Ssum.test == test)]
            if sub.empty:
                continue
            a = sub.set_index("score").aurc
            best = min(a["s1"], a["neg_u"])
            red = 1 - a["fused"] / best
            wins[fam] += red >= 0.10
            lines.append(f"| {LABEL[(task, test)]} | {a['s1']:.3f} | {a['neg_u']:.3f} | {a['fused']:.3f} | "
                         f"{red:+.1%} | {'yes' if red >= 0.10 else 'no'} |")
        h3 = wins["D1"] >= 2 and wins["D3"] >= 2
        lines += ["", f"Wins: D1 {wins['D1']}/3, D3 {wins['D3']}/5 (needs ≥ 2 each). "
                  f"**H3 {'holds' if h3 else 'does not hold'}.**", ""]
        # H4
        lines += ["### H4: conformal coverage (nominal 90%, calibrated on C_r)", "",
                  "| Test | coverage, all queries | coverage, accepted by Gate B (C_h, t = 0.8) |", "|---|---|---|"]
        def cov(task, test, cal="C_r"):
            r = Msum[(Msum.task == task) & (Msum.model == model) & (Msum.cal == cal) &
                     (Msum.test == test) & (Msum.gate == "conformal90")]
            return r.conf_cov_all.iloc[0] if len(r) else np.nan
        def cov_acc(task, test):
            # coverage of C_r-calibrated intervals on the queries Gate B (C_h, 0.8) accepts
            vals = []
            for f in sorted(PRED.glob(f"{task}__{model}__s*.csv.gz")):
                df = pd.read_csv(f)
                df["correct"] = (df.y - df.yhat).abs() <= tol(task)
                seed = int(f.name.split("__s")[1].split(".")[0])
                cr, ch = df[df.set == "C_r"], df[df.set == "C_h"].reset_index(drop=True)
                rng2 = np.random.default_rng(10_000 + seed)
                perm = rng2.permutation(len(ch))
                A, Bh = ch.iloc[perm[: len(ch) // 2]], ch.iloc[perm[len(ch) // 2 :]]
                fz = gates.FusedScorer.fit(A.s1, A.s5, A.u, A.correct)
                th = gates.ltt_threshold(fz.score(Bh.s1, Bh.s5, Bh.u), Bh.correct, 0.8, DELTA)
                q, eps = gates.conformal_quantile(cr.y, cr.yhat, cr.u, 0.1)
                T = df[df.set == test]
                acc = fz.score(T.s1, T.s5, T.u) >= th
                if acc.sum():
                    vals.append(gates.conformal_covered(T.y[acc], T.yhat[acc], T.u[acc], q, eps).mean())
            return float(np.mean(vals)) if vals else np.nan
        rand_ok = True
        for task in sorted({t for _, t, _ in HELD_OUT}):
            c = cov(task, "R")
            if np.isnan(c):
                continue
            rand_ok &= c >= 0.88
            lines.append(f"| {task} / R (random) | {c:.3f} | |")
        below, recovered = 0, True
        for fam, task, test in HELD_OUT:
            c = cov(task, test)
            if np.isnan(c):
                continue
            ca = cov_acc(task, test) if c < 0.85 else np.nan
            if c < 0.85:
                below += 1
                recovered &= (not np.isnan(ca)) and ca >= 0.88
            lines.append(f"| {LABEL[(task, test)]} | {c:.3f} | {'' if np.isnan(ca) else f'{ca:.3f}'} |")
        h4 = rand_ok and below >= 1 and recovered
        lines += ["", f"Random tests ≥ 0.88: {'yes' if rand_ok else 'no'}; held-out tests below 0.85: {below}; "
                  f"recovered to ≥ 0.88 on accepted queries wherever it fell: {'yes' if recovered and below else 'no'}. "
                  f"**H4 {'holds' if h4 else 'does not hold'}.**", ""]
        verdicts[model] = {"H1": bool(h1_all), "H2": bool(h2), "H3": bool(h3), "H4": bool(h4)}
    (OUT / "hypotheses.md").write_text("\n".join(lines))
    (OUT / "hypotheses.json").write_text(json.dumps(verdicts, indent=1))
    print(json.dumps(verdicts, indent=1))


if __name__ == "__main__":
    main()
