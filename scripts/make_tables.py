"""LaTeX tables for the paper, generated from results/ so no number is copied by hand."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyse import HELD_OUT, LABEL
from aoa.reporting import gate_reports

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = ROOT / "paper" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
G = pd.read_csv(RES / "summary_gates.csv")
L = pd.read_csv(RES / "metrics_long.csv")
S = pd.read_csv(RES / "summary_scores.csv")
PER_SEED, POOLED = gate_reports(L, HELD_OUT)
# M3, protocol amendment 2.0: analysed apart, so that version 1's files stay unchanged
A2 = RES / "amendment2"
if (A2 / "summary_gates.csv").exists():
    L2 = pd.read_csv(A2 / "metrics_long.csv")
    PER_SEED2, POOLED2 = gate_reports(L2, HELD_OUT)
    G = pd.concat([G, pd.read_csv(A2 / "summary_gates.csv")], ignore_index=True)
    S = pd.concat([S, pd.read_csv(A2 / "summary_scores.csv")], ignore_index=True)
else:
    PER_SEED2 = POOLED2 = None


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


def pct(x):
    return f"{100*x:.2f}\\%"


def t3(model="RF", target=0.8):
    """Descriptive pooled card with counts; no independent-binomial intervals."""
    out = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
           " & \\multicolumn{3}{c}{$C_r$ pipeline} & \\multicolumn{3}{c}{$C_h$ pipeline} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
           "Held-out test & Coverage & Precision (correct/accepted) & Seeds & Coverage & Precision (correct/accepted) & Seeds \\\\",
           "\\midrule"]
    for fam, task, test in HELD_OUT:
        cells = []
        for cal in ("C_r", "C_h"):
            pooled = POOLED2 if model == "DMPNN" else POOLED
            d = pooled[(pooled.task == task) & (pooled.model == model) & (pooled.cal == cal) &
                       (pooled.test == test) & (pooled.target == target)].iloc[0]
            precision = "abstained (0/0)" if d.n_acc == 0 else f"{d.pooled_precision:.2f} ({int(d.n_correct)}/{int(d.n_acc)})"
            cells += [pct(d.coverage), precision, f"{int(d.seeds_accepting)}/{int(d.seeds)}"]
        out.append(f"{LABEL[(task, test)]} & " + " & ".join(cells) + r" \\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"t3_card_{model}_{int(target*10)}.tex").write_text("\n".join(out))


def per_seed_table(model, target):
    """Both pipelines in the same row; counts refer to prediction-seed observations."""
    head = [r"Test / seed & \multicolumn{3}{c}{$C_r$ pipeline} & \multicolumn{3}{c}{$C_h$ pipeline} \\",
            r" & Coverage & Correct/accepted & Precision & Coverage & Correct/accepted & Precision \\", r"\midrule"]
    out = [r"\begin{longtable}{@{}lrrrrrr@{}}",
           f"\\caption{{Per-seed Gate~B results, {model}, $t={target}$. Coverage is the percentage of test queries accepted. A dash denotes undefined precision when no query is accepted; counts are retained.}}\\\\",
           r"\toprule", *head, r"\endfirsthead", r"\toprule", *head, r"\endhead",
           r"\bottomrule", r"\endfoot"]
    for fam, task, test in HELD_OUT:
        ps = PER_SEED2 if model == "DMPNN" else PER_SEED
        d = ps[(ps.task == task) & (ps.test == test) & (ps.model == model) & (ps.target == target)]
        for seed in sorted(d.seed.unique()):
            cells = []
            for cal in ("C_r", "C_h"):
                row = d[(d.seed == seed) & (d.cal == cal)].iloc[0]
                cells += [pct(row.coverage), f"{int(row.n_correct)}/{int(row.n_acc)}",
                          "---" if row.n_acc == 0 else f"{row.precision:.3f}"]
            out.append(f"{fam}: {LABEL[(task, test)]} / {seed} & " + " & ".join(cells) + r" \\")
        out.append(r"\addlinespace")
    out.append(r"\end{longtable}")
    (TAB / f"t6_per_seed_{model}_{int(target*10)}.tex").write_text("\n".join(out))


def revision_reports():
    dest = RES / "revision"
    dest.mkdir(exist_ok=True)
    PER_SEED.to_csv(dest / "gate_per_seed.csv", index=False)
    POOLED.to_csv(dest / "gate_pooled_descriptive.csv", index=False)
    for model in ("RF", "MLP"):
        for target in (0.8, 0.9):
            per_seed_table(model, target)
    if PER_SEED2 is not None:  # amendment 2.0: M3 reports kept with M3's results
        PER_SEED2.to_csv(A2 / "gate_per_seed.csv", index=False)
        POOLED2.to_csv(A2 / "gate_pooled_descriptive.csv", index=False)
        for target in (0.8, 0.9):
            per_seed_table("DMPNN", target)


def h2_vacuous(model):
    """H2 holds only because the C_h gate abstained (undefined precision counts as meeting t)."""
    meet = cells = 0
    for fam, task, test in HELD_OUT:
        for t in (0.8, 0.9):
            r = G[(G.task == task) & (G.model == model) & (G.cal == "C_h") & (G.test == test) &
                  (G.gate == "LTT_fused") & (G.target == t)]
            if r.empty:
                continue
            cells += 1
            meet += (not np.isnan(r.precision.iloc[0])) and r.precision.iloc[0] >= t - 0.05
    return meet < 0.75 * cells


def t4():
    """Hypothesis verdicts for every model (M3 added by protocol amendment 2.0)."""
    v = json.loads((RES / "hypotheses.json").read_text())
    if (A2 / "hypotheses.json").exists():
        v.update(json.loads((A2 / "hypotheses.json").read_text()))
    cols = [c for c in (("RF", "M1 (RF)"), ("MLP", "M2 (MLP)"), ("DMPNN", "M3 (D-MPNN)")) if c[0] in v]
    out = ["\\begin{tabular}{@{}l" + "l" * len(cols) + "@{}}", "\\toprule",
           "Hypothesis & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    names = {"H1": "H1: gate lowers error at 30\\% coverage", "H2": "H2: calibration split matters",
             "H3": "H3: fusion beats best single signal", "H4": "H4: gating restores conformal coverage"}
    for h in ["H1", "H2", "H3", "H4"]:
        cells = []
        for key, _ in cols:
            word = "holds" if v[key].get(h) else "does not hold"
            if h == "H2" and word == "holds" and h2_vacuous(key):
                word = "holds (vacuously)"
            cells.append(word)
        out.append(f"{names[h]} & " + " & ".join(cells) + " \\\\")
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
    revision_reports(); t1(); t2("RF"); t2("MLP"); t3("RF", 0.8); t3("RF", 0.9); t3("MLP", 0.8); t4(); t5("RF"); t5("MLP")
    if (G.model == "DMPNN").any():  # protocol amendment 2.0
        t2("DMPNN"); t3("DMPNN", 0.8); t3("DMPNN", 0.9); t5("DMPNN")
    print("tables written to", TAB)
