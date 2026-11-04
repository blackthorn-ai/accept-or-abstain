"""LaTeX tables for the paper, generated from results/ so no number is copied by hand."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyse import HELD_OUT, LABEL
from aoa import metrics

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = ROOT / "paper" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
G = pd.read_csv(RES / "summary_gates.csv")
L = pd.read_csv(RES / "metrics_long.csv")
S = pd.read_csv(RES / "summary_scores.csv")


def meta(task):
    ms = [json.loads(f.read_text()) for f in sorted((RES / "predictions").glob(f"{task}__RF__s*.json"))]
    keys = ["n_fit", "n_C_r", "n_C_h", "n_R", "n_H"]
    return {k: int(round(np.mean([m[k] for m in ms]))) for k in keys}


def t1():
    rows = [
        ("LogP (D1)", "logp", "Butina cluster (3,349)", "cluster; E1 (327)"),
        ("LogP, no sp\\textsuperscript{3} C--F (D1)", "logp_noCF", "Butina cluster (3,160)", "cluster; E2 (183)"),
        ("Buchwald--Hartwig Test1--4 (D3a)", "bh_Test1", "additive (17 + 5)", "5 unseen additives"),
        ("Suzuki--Miyaura (D3b)", "sm_ligand", "ligand (12)", "2 unseen ligands"),
    ]
    out = ["\\begin{tabular}{@{}llrrrrrl@{}}", "\\toprule",
           "Task & Groups (number) & $F$ & $C_r$ & $C_h$ & $R$ & $H$ & Held-out tests \\\\", "\\midrule"]
    for name, task, grp, tests in rows:
        m = meta(task)
        out.append(f"{name} & {grp} & {m['n_fit']:,} & {m['n_C_r']:,} & {m['n_C_h']:,} & {m['n_R']:,} & {m['n_H']:,} & {tests} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "t1_splits.tex").write_text("\n".join(out))


def t2(model="RF"):
    items = [("logp", "R"), ("logp", "H"), ("logp", "E1"), ("logp_noCF", "E2"), ("logp_noCF", "E2_CF"),
             ("logp_noCF", "E2_parent"),
             *[(f"bh_Test{i}", s) for i in range(1, 5) for s in ("R", "H")], ("sm_ligand", "R"), ("sm_ligand", "H")]
    lab = {("logp", "R"): "LogP, random $R$", ("logp", "H"): "LogP, held-out cluster",
           ("logp", "E1"): "LogP, prospective E1", ("logp_noCF", "E2"): "LogP, fluorinated E2 (all)",
           ("logp_noCF", "E2_CF"): "\\quad sp\\textsuperscript{3} C--F compounds",
           ("logp_noCF", "E2_parent"): "\\quad parents", ("sm_ligand", "R"): "SM, random $R$",
           ("sm_ligand", "H"): "SM, held-out ligands"}
    for i in range(1, 5):
        lab[(f"bh_Test{i}", "R")] = f"BH Test{i}, random $R$"
        lab[(f"bh_Test{i}", "H")] = f"BH Test{i}, held-out additives"
    out = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
           "Test set & $n$ & RMSE & $R^2$ & Error rate \\\\", "\\midrule"]
    for k in items:
        r = G[(G.task == k[0]) & (G.test == k[1]) & (G.model == model) & (G.cal == "C_h") & (G.gate == "none")]
        if r.empty:
            continue
        r = r.iloc[0]
        if k in [("logp", "R"), ("bh_Test1", "R"), ("sm_ligand", "R")] and k != ("logp", "R"):
            out.append("\\addlinespace")
        out.append(f"{lab[k]} & {int(r.n):,} & {r.rmse:.2f} $\\pm$ {r.rmse_sd:.2f} & {r.r2:.2f} $\\pm$ {r.r2_sd:.2f} & {1-r.precision:.2f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"t2_accuracy_{model}.tex").write_text("\n".join(out))


def t3(model="RF", target=0.8):
    """Operating-limits card: pooled precision over all accepted queries of all seeds, Wilson 95% CI."""
    out = ["\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
           " & \\multicolumn{2}{c}{Calibrated on $C_r$} & \\multicolumn{3}{c}{Calibrated on $C_h$} \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}",
           "Held-out test & Coverage & Precision [95\\% CI] & Coverage & Precision [95\\% CI] & Seeds accepting \\\\",
           "\\midrule"]
    for fam, task, test in HELD_OUT:
        cells = []
        for cal in ("C_r", "C_h"):
            d = L[(L.task == task) & (L.model == model) & (L.cal == cal) & (L.test == test) &
                  (L.gate == "LTT_fused") & (L.target == target)]
            n = int(d.n_acc.sum())
            k = int(round((d.precision.fillna(0) * d.n_acc).sum()))
            cov = d.coverage.mean()
            if n == 0:
                cells.append((f"{cov:.2f}", "abstained", "0/5"))
            else:
                lo, hi = metrics.wilson(k, n)
                cells.append((f"{cov:.2f}", f"{k/n:.2f} [{lo:.2f}, {hi:.2f}]", f"{int((d.n_acc > 0).sum())}/{len(d)}"))
        (cr_cov, cr_p, _), (ch_cov, ch_p, ch_s) = cells
        out.append(f"{LABEL[(task, test)]} & {cr_cov} & {cr_p} & {ch_cov} & {ch_p} & {ch_s} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"t3_card_{model}_{int(target*10)}.tex").write_text("\n".join(out))


def t4():
    """Hypothesis verdicts for both models."""
    v = json.loads((RES / "hypotheses.json").read_text())
    out = ["\\begin{tabular}{@{}lll@{}}", "\\toprule", "Hypothesis & M1 (RF) & M2 (MLP) \\\\", "\\midrule"]
    names = {"H1": "H1: gate lowers error at 30\\% coverage", "H2": "H2: calibration split matters",
             "H3": "H3: fusion beats best single signal", "H4": "H4: gating restores conformal coverage"}
    for h in ["H1", "H2", "H3", "H4"]:
        mlp = "holds" if v.get("MLP", {}).get(h) else "does not hold"
        if h == "H2" and mlp == "holds":
            mlp = "holds (vacuously)"
        out.append(f"{names[h]} & {'holds' if v['RF'][h] else 'does not hold'} & {mlp} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "t4_verdicts.tex").write_text("\n".join(out))


def t5(model):
    """AURC per score."""
    out = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
           "Held-out test & No gate & $s_1$ & $-u$ & Fused \\\\", "\\midrule"]
    for fam, task, test in HELD_OUT:
        sub = S[(S.task == task) & (S.model == model) & (S.cal == "C_h") & (S.test == test)]
        if sub.empty:
            continue
        a = sub.set_index("score")
        out.append(f"{LABEL[(task, test)]} & {a.err_all.iloc[0]:.3f} & {a.aurc['s1']:.3f} & {a.aurc['neg_u']:.3f} & {a.aurc['fused']:.3f} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"t5_aurc_{model}.tex").write_text("\n".join(out))


if __name__ == "__main__":
    t1(); t2("RF"); t2("MLP"); t3("RF", 0.8); t3("RF", 0.9); t3("MLP", 0.8); t4(); t5("RF"); t5("MLP")
    print("tables written to", TAB)
