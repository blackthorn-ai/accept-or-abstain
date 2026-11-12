"""Stage 3: figures and LaTeX tables for the paper, from results/ (run analyse.py first)."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from analyse import HELD_OUT, LABEL
from aoa import metrics

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
# colour-blind-safe (Okabe-Ito)
C = {"s1": "#0072B2", "neg_u": "#E69F00", "fused": "#009E73", "none": "#7f7f7f",
     "C_r": "#D55E00", "C_h": "#009E73", "R": "#9ecae1", "H": "#3182bd"}
NAME = {"s1": "Similarity s1", "neg_u": "Ensemble spread u", "fused": "Fused (Gate B score)"}
MODEL = "RF"

G = pd.read_csv(RES / "summary_gates.csv")
S = pd.read_csv(RES / "summary_scores.csv")
with open(RES / "cache" / "arrays.pkl", "rb") as fh:
    A = pickle.load(fh)


def gsel(**kw):
    d = G
    for k, v in kw.items():
        d = d[d[k] == v] if not (isinstance(v, float) and np.isnan(v)) else d[d[k].isna()]
    return d


# ------------------------------------------------------------------ F1 protocol
def fig_protocol():
    steps = [
        ("P1", "Context of use;\ncorrect if\n|error| \u2264 \u03c4"),
        ("P2", "Split by\nchemistry:\n$F$, $C_h$, $H$"),
        ("P3", "Fit model on $F$;\nsignals\n$s_1$, $s_5$, $u$"),
        ("P4", "Calibrate gate\non $C_h$\n(empirical, target $t$)"),
        ("P5", "Stress-test on\nfurther\nheld-out sets"),
        ("P6", "Publish the\noperating-\nlimits card"),
    ]
    w, gap, y0, h, x0 = 14.6, 1.9, 7, 19, 1.4
    for fs in (7.0, 6.7, 6.4, 6.1, 5.8):
        fig, ax = plt.subplots(figsize=(7.1, 2.1))
        fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
        ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")
        pairs = []
        for i, (code, txt) in enumerate(steps):
            x = x0 + i * (w + gap)
            box = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                 fc="#eef5f0" if code == "P4" else "#f3f6fa",
                                 ec="#009E73" if code == "P4" else "#4a5a70", lw=1.0)
            ax.add_patch(box)
            ax.text(x + w / 2, y0 + h - 3.4, code, ha="center", va="center", weight="bold", fontsize=8.5)
            body = ax.text(x + w / 2, y0 + h / 2 - 2.6, txt, ha="center", va="center",
                           fontsize=fs, linespacing=1.2)
            pairs.append((box, body))
            if i < len(steps) - 1:
                ax.add_patch(FancyArrowPatch((x + w + 0.35, y0 + h / 2), (x + w + gap - 0.35, y0 + h / 2),
                                             arrowstyle="-|>", mutation_scale=8, lw=1.0, color="#4a5a70"))
        foot = ax.text(50, 2.2, "Accept or decline each query; achieved precision and coverage are empirical, "
                       "with no established guarantee.",
                       ha="center", fontsize=7.0, style="italic", color="#333")
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        margin = 0.06 * fig.dpi  # about 0.06 inch of clearance inside each box
        ax_bb = ax.get_window_extent(r)
        fits = all(
            bb.x0 >= b.x0 + margin and bb.x1 <= b.x1 - margin and bb.y0 >= b.y0 + margin
            and b.x0 >= ax_bb.x0 and b.x1 <= ax_bb.x1
            for b, bb in ((bx.get_window_extent(r), tx.get_window_extent(r)) for bx, tx in pairs)
        ) and foot.get_window_extent(r).x1 <= ax_bb.x1 and foot.get_window_extent(r).x0 >= ax_bb.x0
        if fits:
            break
        plt.close(fig)
    assert fits, "Figure 1 labels do not fit their boxes"
    fig.savefig(FIG / "fig1_protocol.pdf"); fig.savefig(FIG / "fig1_protocol.png")
    plt.close(fig)
    print(f"fig1: body font {fs} pt fits all boxes")


# ------------------------------------------------------------------ F2 the gap
def fig_gap():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), gridspec_kw={"width_ratios": [1, 1.5]})
    # LogP
    ax = axes[0]
    items = [("logp", "R", "Random"), ("logp", "H", "Cluster"), ("logp", "E1", "Prosp. E1"),
             ("logp_noCF", "E2", "Fluor. E2")]
    vals = [gsel(task=t, model=MODEL, cal="C_h", test=s, gate="none") for t, s, _ in items]
    m = [v.rmse.iloc[0] for v in vals]; sd = [v.rmse_sd.iloc[0] for v in vals]
    ax.bar(range(4), m, yerr=sd, color=[C["R"], C["H"], C["H"], C["H"]], capsize=2, width=0.65)
    ax.set_xticks(range(4), [x[2] for x in items], rotation=0)
    ax.set_ylabel("RMSE (log units)"); ax.set_title("LogP")
    for i, v in enumerate(m):
        ax.text(i, v + sd[i] + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    # yields
    ax = axes[1]
    tasks = ["bh_Test1", "bh_Test2", "bh_Test3", "bh_Test4", "sm_ligand"]
    lab = ["BH T1", "BH T2", "BH T3", "BH T4", "SM lig."]
    x = np.arange(len(tasks))
    for j, (s, col, off) in enumerate([("R", C["R"], -0.18), ("H", C["H"], 0.18)]):
        v = [gsel(task=t, model=MODEL, cal="C_h", test=s, gate="none") for t in tasks]
        mm = [r.rmse.iloc[0] for r in v]; ss = [r.rmse_sd.iloc[0] for r in v]
        ax.bar(x + off, mm, yerr=ss, width=0.36, color=col, capsize=2,
               label="Random test R" if s == "R" else "Held-out test H")
    ax.set_xticks(x, lab); ax.set_ylabel("RMSE (yield, %)"); ax.set_title("Reaction yield")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_gap.pdf"); fig.savefig(FIG / "fig2_gap.png"); plt.close(fig)


# ------------------------------------------------------------------ F3 risk-coverage
def pooled_curve(task, test, score, cal="C_h", grid=np.linspace(0.02, 1, 50)):
    curves = []
    for (tk, m, s, c, te, k), d in A.items():
        if tk == task and m == MODEL and c == cal and te == test and k == score:
            cov, risk = metrics.risk_coverage(d["score"], d["correct"])
            curves.append(np.interp(grid, cov, risk))
    return grid, np.mean(curves, 0), np.std(curves, 0)


def fig_risk_coverage():
    panels = [("logp", "H"), ("logp", "E1"), ("logp_noCF", "E2"),
              ("bh_Test2", "H"), ("bh_Test4", "H"), ("sm_ligand", "H")]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.4), sharex=True)
    for ax, (task, test) in zip(axes.flat, panels):
        for k in ["s1", "neg_u", "fused"]:
            g, mu, sd = pooled_curve(task, test, k)
            ax.plot(g, mu, color=C[k], lw=1.4, label=NAME[k])
            ax.fill_between(g, mu - sd, mu + sd, color=C[k], alpha=0.12, lw=0)
        err_all = S[(S.task == task) & (S.model == MODEL) & (S.cal == "C_h") & (S.test == test) &
                    (S.score == "fused")].err_all.iloc[0]
        ax.axhline(err_all, color=C["none"], ls="--", lw=1, label="No gate")
        ax.axvline(0.3, color="#bbb", lw=0.6, ls=":")
        ax.set_title(LABEL[(task, test)]); ax.set_ylim(0, min(1, max(0.25, err_all * 1.35)))
    for ax in axes[1]:
        ax.set_xlabel("Coverage")
    for ax in axes[:, 0]:
        ax.set_ylabel("Selective error rate")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7.2, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "fig3_risk_coverage.pdf"); fig.savefig(FIG / "fig3_risk_coverage.png"); plt.close(fig)


# ------------------------------------------------------------------ F4 calibration split
def fig_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharey=True)
    for ax, t in zip(axes, [0.8, 0.9]):
        for j, cal in enumerate(["C_r", "C_h"]):
            xs, ys, cv = [], [], []
            for i, (fam, task, test) in enumerate(HELD_OUT):
                r = gsel(task=task, model=MODEL, cal=cal, test=test, gate="LTT_fused", target=t)
                if r.empty:
                    continue
                xs.append(i + (-0.15 if cal == "C_r" else 0.15)); ys.append(r.precision.iloc[0]); cv.append(r.coverage.iloc[0])
            ax.scatter(xs, ys, s=[18 + 80 * c for c in cv], color=C[cal], alpha=0.85, edgecolor="white", lw=0.5,
                       label="Scorer + threshold on random C_r" if cal == "C_r" else "Scorer + threshold on held-out C_h", zorder=3)
        ax.axhline(t, color="k", lw=0.8); ax.axhspan(t - 0.05, t, color="#e5e5e5", zorder=0)
        ax.set_xticks(range(len(HELD_OUT)), [LABEL[(tk, te)].replace(" (", "\n(").replace("LogP ", "LogP\n") for _, tk, te in HELD_OUT],
                      rotation=60, ha="right", fontsize=6.8)
        ax.set_title(f"Target precision t = {t}"); ax.set_ylim(0.35, 1.02)
    axes[0].set_ylabel("Achieved precision on the held-out test")
    axes[0].legend(fontsize=6.8, loc="lower left")
    fig.text(0.99, 0.01, "Marker area ∝ coverage", ha="right", fontsize=6.5, color="#555")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_calibration.pdf"); fig.savefig(FIG / "fig4_calibration.png"); plt.close(fig)


# ------------------------------------------------------------------ F5 conformal
def fig_conformal():
    rows = []
    for fam, task, test in [("", "logp", "R"), *HELD_OUT[:3], ("", "bh_Test1", "R"), *HELD_OUT[3:]]:
        allc, accc = [], []
        for (tk, m, s, c, te, k), d in A.items():
            if tk == task and m == MODEL and te == test and k == "fused" and c == "C_r":
                allc.append(d["covered"].mean())
        # coverage of C_r intervals on queries accepted by the C_h gate
        for (tk, m, s, c, te, k), d in A.items():
            if tk == task and m == MODEL and te == test and k == "fused" and c == "C_h":
                cr = A[(tk, m, s, "C_r", te, k)]["covered"]
                if d["acc08"].sum():
                    accc.append(cr[d["acc08"]].mean())
        lab = (LABEL.get((task, test)) or f"{'LogP' if task=='logp' else 'BH Test1'} random R")
        rows.append((lab, np.mean(allc), np.mean(accc) if accc else np.nan))
    fig, ax = plt.subplots(figsize=(7.1, 2.4))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r[1] for r in rows], 0.36, color="#9aa6b2", label="All queries")
    ax.bar(x + 0.18, [r[2] for r in rows], 0.36, color=C["fused"], label="Accepted by Gate B (C_h, t = 0.8)")
    ax.axhline(0.9, color="k", lw=0.8); ax.axhline(0.85, color="k", lw=0.6, ls=":")
    ax.set_xticks(x, [r[0].replace(" (", "\n(").replace(" random R", "\nrandom R").replace("LogP ", "LogP\n").replace("BH ", "BH\n").replace("SM ", "SM\n") for r in rows], fontsize=6.2)
    ax.set_ylabel("Interval coverage"); ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=7, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "fig5_conformal.pdf"); fig.savefig(FIG / "fig5_conformal.png"); plt.close(fig)
    return rows


# ------------------------------------------------------------------ F6 the fluorinated case
def fig_fluorinated():
    seeds = sorted({s for (tk, m, s, c, te, k) in A if tk == "logp_noCF" and m == MODEL})
    d0 = A[("logp_noCF", MODEL, seeds[0], "C_h", "E2", "fused")]
    cf = d0["sp3_CF"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    acc = d0["acc08"]
    for mask, col, lab in [(~cf, "#56B4E9", "Parent (no sp³ C–F)"), (cf, "#CC79A7", "sp³ C–F compound")]:
        ax.scatter(d0["y"][mask & acc], d0["yhat"][mask & acc], s=16, color=col, label=f"{lab}, accepted", edgecolor="k", lw=0.3)
        ax.scatter(d0["y"][mask & ~acc], d0["yhat"][mask & ~acc], s=16, facecolor="none", edgecolor=col, lw=0.8,
                   label=f"{lab}, declined")
    lo, hi = min(d0["y"].min(), d0["yhat"].min()) - 0.2, max(d0["y"].max(), d0["yhat"].max()) + 0.2
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.6); ax.fill_between([lo, hi], [lo - 0.5, hi - 0.5], [lo + 0.5, hi + 0.5], color="#eee", zorder=0)
    ax.set_xlabel("Measured logP"); ax.set_ylabel("Predicted logP"); ax.set_title(f"Fluorinated hold-out, seed {seeds[0]}")
    ax.legend(fontsize=6.2, loc="upper left")
    ax = axes[1]
    stats = []
    for grp, mask_fn in [("Parents", lambda d: ~d["sp3_CF"]), ("sp³ C–F", lambda d: d["sp3_CF"])]:
        accr, precs, errs = [], [], []
        for s in seeds:
            d = A[("logp_noCF", MODEL, s, "C_h", "E2", "fused")]
            mk = mask_fn(d)
            accr.append(d["acc08"][mk].mean())
            errs.append(1 - d["correct"][mk].mean())
            a = d["acc08"] & mk
            precs.append(d["correct"][a].mean() if a.sum() else np.nan)
        stats.append((grp, np.mean(errs), np.mean(accr), np.nanmean(precs) if not np.all(np.isnan(precs)) else np.nan))
    x = np.arange(2)
    ax.bar(x - 0.2, [s[1] for s in stats], 0.4, color="#9aa6b2", label="Ungated error rate")
    ax.bar(x + 0.2, [s[2] for s in stats], 0.4, color=C["fused"], label="Share accepted (t = 0.8)")
    ax.set_xticks(x, [s[0] for s in stats]); ax.set_ylim(0, 1); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Mean over five seeds")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_fluorinated.pdf"); fig.savefig(FIG / "fig6_fluorinated.png"); plt.close(fig)
    return stats


# ------------------------------------------------------------------ F7 signals by learner (amendment 2.0)
def fig_signals_by_learner():
    """AURC of each score on every held-out test, for each learner (fusion fitted on C_h)."""
    a2 = RES / "amendment2" / "summary_scores.csv"
    SS = pd.concat([S, pd.read_csv(a2)], ignore_index=True) if a2.exists() else S
    learners = [("RF", "M1, random forest"), ("MLP", "M2, MLP ensemble"), ("DMPNN", "M3, D-MPNN ensemble")]
    learners = [m for m in learners if (SS.model == m[0]).any()]
    fig, axes = plt.subplots(1, len(learners), figsize=(7.1, 2.9), sharey=True, sharex=True)
    y = np.arange(len(HELD_OUT))[::-1]
    for ax, (mk, title) in zip(np.atleast_1d(axes), learners):
        for sc, mark, off in [("s1", "o", 0.2), ("neg_u", "s", 0.0), ("fused", "D", -0.2)]:
            vals = []
            for _, task, test in HELD_OUT:
                r = SS[(SS.task == task) & (SS.model == mk) & (SS.cal == "C_h") & (SS.test == test) & (SS.score == sc)]
                vals.append(r.aurc.iloc[0] if len(r) else np.nan)
            ax.scatter(vals, y + off, s=14, marker=mark, color=C[sc], label=NAME[sc], zorder=3, lw=0)
        ax.axhline(4.5, color="0.6", lw=0.6)
        ax.grid(axis="x", lw=0.4, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("AURC (lower is better)")
    axes[0].set_yticks(y, [LABEL[(t, s)] for _, t, s in HELD_OUT])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.55, 1.07))
    fig.tight_layout()
    fig.savefig(FIG / "fig7_signals_by_learner.pdf"); fig.savefig(FIG / "fig7_signals_by_learner.png"); plt.close(fig)


if __name__ == "__main__":
    fig_protocol(); fig_gap(); fig_risk_coverage(); fig_calibration()
    if (RES / "amendment2" / "summary_scores.csv").exists():  # protocol amendment 2.0
        fig_signals_by_learner()
    conf = fig_conformal(); fl = fig_fluorinated()
    pd.DataFrame(conf, columns=["test", "cov_all", "cov_accepted"]).to_csv(RES / "fig5_values.csv", index=False)
    pd.DataFrame(fl, columns=["group", "err_rate", "accepted_share", "precision_accepted"]).to_csv(RES / "fig6_values.csv", index=False)
    print("figures written to", FIG)


# ------------------------------------------------------------------ graphical abstract (ChemRxiv)
def _molecule_png(smiles: str, w: int = 330, h: int = 230):
    """RDKit depiction with a transparent background, thick bonds for small display."""
    import io
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions()
    o.clearBackground = False
    o.bondLineWidth = 6
    o.padding = 0.06
    o.minFontSize = 30
    o.maxFontSize = 34
    d.DrawMolecule(Chem.MolFromSmiles(smiles))
    d.FinishDrawing()
    return plt.imread(io.BytesIO(d.GetDrawingText()), format="png")


def fig_graphical_abstract():
    L = pd.read_csv(RES / "metrics_long.csv")
    L = L[(L.model == MODEL) & (L.gate == "LTT_fused") & (L.target == 0.8) & (L.test == "H")]
    tests = ["bh_Test1", "bh_Test2", "bh_Test3", "bh_Test4"]
    pooled = {}
    for cal in ("C_r", "C_h"):
        for t in tests:
            d = L[(L.task == t) & (L.cal == cal)]
            n = d.n_acc.sum()
            pooled[(cal, t)] = ((d.precision.fillna(0) * d.n_acc).sum() / n if n else np.nan, d.coverage.mean())

    ink, grey = "#1f2a36", "#5f6b78"
    fig = plt.figure(figsize=(7.0, 3.5))
    A = fig.add_axes([0, 0, 1, 1]); A.set_xlim(0, 100); A.set_ylim(0, 50); A.axis("off")
    fig.text(0.5, 0.925, "Compare abstention pipelines on held-out chemistry", ha="center", va="center",
             fontsize=12.0, weight="bold", color=ink)

    # ---- left: query -> gate -> prediction / abstain
    ax_m = fig.add_axes([0.0, 0.39, 0.175, 0.35]); ax_m.imshow(_molecule_png("c1ccc(-c2ccno2)cc1")); ax_m.axis("off")
    A.text(9.0, 18.6, "unseen\nadditive", ha="center", va="center", fontsize=10.5, color=grey, linespacing=1.1)
    A.add_patch(FancyArrowPatch((17.2, 28.5), (20.6, 28.5), arrowstyle="-|>", mutation_scale=13, lw=1.6, color=ink))
    A.add_patch(FancyBboxPatch((21.2, 22.5), 10.6, 12, boxstyle="round,pad=0.4,rounding_size=1.6",
                               fc="#ffffff", ec=ink, lw=1.6))
    A.text(26.5, 30.6, "gate", ha="center", va="center", fontsize=13, weight="bold", color=ink)
    A.text(26.5, 25.6, "$s_1$, $u$", ha="center", va="center", fontsize=11, color=grey)
    A.add_patch(FancyArrowPatch((32.6, 31.5), (36.0, 35.0), arrowstyle="-|>", mutation_scale=12, lw=1.5, color=ink))
    A.add_patch(FancyArrowPatch((32.6, 25.5), (36.0, 22.0), arrowstyle="-|>", mutation_scale=12, lw=1.5, color=ink))
    A.text(36.6, 35.6, "\u2713 predict", ha="left", va="center", fontsize=12, color=ink)
    A.text(36.6, 21.4, "\u2717 abstain", ha="left", va="center", fontsize=12, color=ink)
    # ---- calibration choice doubles as the chart legend
    A.text(2.0, 12.2, "Scorer + threshold fitted on", ha="left", va="center", fontsize=10.5, color=grey)
    A.add_patch(FancyArrowPatch((26.5, 14.4), (26.5, 21.6), arrowstyle="-|>", mutation_scale=11, lw=1.3,
                                color=grey, linestyle=(0, (3, 2))))
    for y, cal, lab in [(7.0, "C_r", "a random fold"), (2.6, "C_h", "held-out chemistry")]:
        A.scatter([3.4], [y], s=95, color=C[cal], zorder=3)
        A.text(5.4, y, lab, ha="left", va="center", fontsize=12, color=C[cal], weight="bold")

    # ---- right: dumbbell chart of achieved precision at target 0.8
    ax = fig.add_axes([0.615, 0.155, 0.365, 0.645])
    x = np.arange(len(tests))
    for xi, t in zip(x, tests):
        (pr, cr), (ph, ch) = pooled[("C_r", t)], pooled[("C_h", t)]
        if ph - pr > 0.06:  # a shorter arrow would be hidden behind the dots
            ax.annotate("", xy=(xi, ph), xytext=(xi, pr),
                        arrowprops=dict(arrowstyle="-|>", color="#8a96a3", lw=1.8, shrinkA=7, shrinkB=7,
                                        mutation_scale=12))
        ax.scatter([xi], [pr], s=150, color=C["C_r"], zorder=3)
        ax.scatter([xi], [ph], s=150, color=C["C_h"], zorder=3)
        ax.text(xi + 0.19, pr, f"{cr:.0%}", ha="left", va="center", fontsize=10.5, color=C["C_r"])
        ax.text(xi + 0.19, ph, f"{ch:.0%}", ha="left", va="center", fontsize=10.5, color=C["C_h"])
    ax.axhline(0.8, color=ink, lw=1.4, ls=(0, (5, 3)), zorder=1)
    ax.text(3.72, 0.808, "target", ha="right", va="bottom", fontsize=10.5, color=ink)
    ax.set_xlim(-0.45, 3.75); ax.set_ylim(0.52, 0.93)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9]); ax.tick_params(labelsize=11, colors=ink)
    ax.set_xticks(x, ["1", "2", "3", "4"])
    ax.set_xlabel("Buchwald\u2013Hartwig test (unseen additives)", fontsize=11, color=ink)
    ax.set_ylabel("pooled descriptive precision", fontsize=11, color=ink, linespacing=1.1)
    ax.text(-0.28, 0.915, "% = reactions accepted", ha="left", va="center", fontsize=10, color=grey)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"graphical_abstract.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return pooled


if __name__ == "__main__" and "--ga" in __import__("sys").argv:
    print(fig_graphical_abstract())
