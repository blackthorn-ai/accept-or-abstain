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
        ("P4", "Calibrate gate\non $C_h$\n(LTT, target $t$)"),
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
        foot = ax.text(50, 2.2, "Each query is accepted, with a prediction, or declined; the precision "
                       "target holds for queries exchangeable with $C_h$.",
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
                       label="Calibrated on random fold C_r" if cal == "C_r" else "Calibrated on held-out fold C_h", zorder=3)
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


if __name__ == "__main__":
    fig_protocol(); fig_gap(); fig_risk_coverage(); fig_calibration()
    conf = fig_conformal(); fl = fig_fluorinated()
    pd.DataFrame(conf, columns=["test", "cov_all", "cov_accepted"]).to_csv(RES / "fig5_values.csv", index=False)
    pd.DataFrame(fl, columns=["group", "err_rate", "accepted_share", "precision_accepted"]).to_csv(RES / "fig6_values.csv", index=False)
    print("figures written to", FIG)
