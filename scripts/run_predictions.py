"""Stage 1: fit the base models and write per-query predictions and signals.

Usage: python scripts/run_predictions.py [--tasks logp,logp_noCF,bh,sm] [--models RF,MLP]

Writes results/predictions/<task>__<model>__s<seed>.csv.gz, one row per query in
every non-fit set (C_r, C_h, R, H and any external set). Nothing here looks at gates.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aoa import data, features, splits
from aoa.models import MODELS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
PRED = ROOT / "results" / "predictions"
SEEDS = [0, 1, 2, 3, 4]


def cached(name, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return tuple(z[k] for k in sorted(z.files))
    out = fn()
    np.savez_compressed(f, **{f"a{i}": a for i, a in enumerate(out)})
    return out


# ---------------------------------------------------------------- task builders
def logp_inputs():
    e1 = data.load_prospective()
    e2 = data.load_ex12()
    pool = data.load_logp_pool(set(e1.inchikey) | set(e2.inchikey))
    Xp, Bp = cached("logp_pool_feat", lambda: features.molecule_features(pool.smiles.tolist()))
    (labels,) = cached("logp_pool_butina065", lambda: (features.butina_clusters(Bp, 0.65),))
    X1, B1 = cached("logp_e1_feat", lambda: features.molecule_features(e1.smiles.tolist()))
    X2, B2 = cached("logp_e2_feat", lambda: features.molecule_features(e2.smiles.tolist()))
    return pool, Xp, Bp, labels, (e1, X1, B1), (e2, X2, B2)


def logp_task(no_cf: bool):
    pool, Xp, Bp, labels, ext1, ext2 = logp_inputs()
    keep = np.flatnonzero(~pool.sp3_CF.values) if no_cf else np.arange(len(pool))
    ext = {"E2": ext2} if no_cf else {"E1": ext1}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        H, Ch, I = splits.group_holdout(labels[keep], rng)
        H, Ch, I = keep[H], keep[Ch], keep[I]
        F, Cr, R = splits.random_part(I, rng)
        sets = {"C_r": Cr, "C_h": Ch, "R": R, "H": H}
        yield seed, dict(X=Xp, B=Bp, y=pool.y.values, F=F, sets=sets, ext=ext,
                         smiles=pool.smiles.values,
                         meta={"n_clusters": int(len(np.unique(labels[keep])))})


def bh_task(sheet: str):
    df, n_train = data.load_bh(sheet)
    X, B = cached(f"bh_{sheet}_feat", lambda: features.reaction_features(df, data.BH_COMPONENTS))
    train = np.arange(n_train)
    H = np.arange(n_train, len(df))
    add = df["Additive"].values
    assert not set(add[train]) & set(add[H]), "additives leak between train and test"
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        _, Ch_loc, I_loc, _, cal_add = splits.named_holdout(add[train], rng, n_test=0, n_cal=4)
        Ch, I = train[Ch_loc], train[I_loc]
        F, Cr, R = splits.random_part(I, rng)
        sets = {"C_r": Cr, "C_h": Ch, "R": R, "H": H}
        yield seed, dict(X=X, B=B, y=df.y.values, F=F, sets=sets, ext={},
                         smiles=data.joined_smiles(df, data.BH_COMPONENTS),
                         meta={"cal_additives": cal_add,
                               "test_additives": sorted(set(add[H]))})


def sm_task():
    df = data.load_sm()
    X, B = cached("sm_feat", lambda: features.reaction_features(df, data.SM_COMPONENTS))
    lig = df["ligand_name"].values
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        H, Ch, I, h_g, c_g = splits.named_holdout(lig, rng, n_test=2, n_cal=2)
        F, Cr, R = splits.random_part(I, rng)
        sets = {"C_r": Cr, "C_h": Ch, "R": R, "H": H}
        yield seed, dict(X=X, B=B, y=df.y.values, F=F, sets=sets, ext={},
                         smiles=data.joined_smiles(df, data.SM_COMPONENTS),
                         meta={"test_ligands": [str(x) for x in h_g],
                               "cal_ligands": [str(x) for x in c_g],
                               "ligand": lig})


TASKS = {
    "logp": lambda: logp_task(False),
    "logp_noCF": lambda: logp_task(True),
    "bh_Test1": lambda: bh_task("Test1"),
    "bh_Test2": lambda: bh_task("Test2"),
    "bh_Test3": lambda: bh_task("Test3"),
    "bh_Test4": lambda: bh_task("Test4"),
    "sm_ligand": sm_task,
}


# ---------------------------------------------------------------- runner
def run_one(task: str, model_name: str, seed: int, spec: dict):
    out_f = PRED / f"{task}__{model_name}__s{seed}.csv.gz"
    if out_f.exists():
        return
    X, B, y, F = spec["X"], spec["B"], spec["y"], spec["F"]
    t0 = time.time()
    model = MODELS[model_name](seed).fit(X[F], y[F])
    rows = []
    blocks = [(name, X[idx], B[idx], y[idx], idx) for name, idx in spec["sets"].items()]
    for name, (edf, Xe, Be) in spec["ext"].items():
        blocks.append((name, Xe, Be, edf.y.values, np.arange(len(edf))))
    for name, Xs, Bs, ys, idx in blocks:
        yhat, u = model.predict(Xs)
        s1, s5 = features.tanimoto_topk(Bs, B[F], k=5)
        part = pd.DataFrame({"set": name, "idx": idx, "y": ys, "yhat": yhat, "u": u, "s1": s1, "s5": s5})
        if name == "E2":
            edf = spec["ext"]["E2"][0]
            part["sp3_CF"] = edf.sp3_CF.values
            part["f_group"] = edf.f_group.values
        if "ligand" in spec["meta"] and name not in spec["ext"]:
            part["group"] = spec["meta"]["ligand"][idx]
        rows.append(part)
    PRED.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(out_f, index=False, compression="gzip")
    meta = {k: v for k, v in spec["meta"].items() if k != "ligand"}
    meta.update({"n_fit": int(len(F)), **{f"n_{k}": int(len(v)) for k, v in spec["sets"].items()},
                 "seconds": round(time.time() - t0, 1)})
    (PRED / f"{task}__{model_name}__s{seed}.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"{task:10s} {model_name:4s} seed {seed}: fit {len(F)}  {meta['seconds']} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--models", default="RF,MLP")
    a = ap.parse_args()
    for task in a.tasks.split(","):
        for seed, spec in TASKS[task]():
            for m in a.models.split(","):
                run_one(task, m, seed, spec)


if __name__ == "__main__":
    main()
