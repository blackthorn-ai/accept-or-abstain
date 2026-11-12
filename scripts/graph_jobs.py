"""Model M3 (protocol amendment 2.0): hand the splits to the graph environment and take back its predictions.

Chemprop needs an older RDKit than the main environment, so M3 is fitted in a separate
environment (.venv-graph, versions in env/graph-requirements.txt). Everything else (data,
splits, similarity signals, gates, metrics) stays in the main environment, exactly as for
M1 and M2. Three steps, run from the repository root:

  python scripts/graph_jobs.py export            # one job file per task and seed
  python scripts/graph_jobs.py train [--workers 6 --threads 3]
                                                 # runs scripts/graph_train.py in .venv-graph
  python scripts/graph_jobs.py merge             # results/predictions_m3/<task>__DMPNN__s<seed>.csv.gz

Export writes every row of F and of each evaluation set with its SMILES and label (reactions:
the component SMILES joined by '.'). Merge adds s1 and s5 the way run_predictions.py does and
checks that they equal the values in the M1 file of the same task and seed. M3 predictions are
kept apart from version 1's (results/predictions_m3/), so the version 1 files and the results
built from them stay byte-identical.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from aoa import features
from run_predictions import PRED, ROOT, TASKS

JOBS = ROOT / "results" / "graph_jobs"
PRED_M3 = ROOT / "results" / "predictions_m3"
GRAPH_PY = ROOT / ".venv-graph" / "bin" / "python"
MODEL = "DMPNN"
# largest jobs first, so that the parallel run ends evenly
ORDER = ["logp", "logp_noCF", "sm_ligand", "bh_Test1", "bh_Test2", "bh_Test3", "bh_Test4"]


def export(tasks: list[str]) -> None:
    JOBS.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        for seed, spec in TASKS[task]():
            f = JOBS / f"{task}__s{seed}.csv.gz"
            if f.exists():
                continue
            smi, y, F = spec["smiles"], spec["y"], spec["F"]
            parts = [pd.DataFrame({"set": "F", "idx": F, "smiles": smi[F], "y": y[F]})]
            for name, idx in spec["sets"].items():
                parts.append(pd.DataFrame({"set": name, "idx": idx, "smiles": smi[idx], "y": y[idx]}))
            for name, (edf, _, _) in spec["ext"].items():
                parts.append(pd.DataFrame({"set": name, "idx": np.arange(len(edf)),
                                           "smiles": edf.smiles.values, "y": edf.y.values}))
            job = pd.concat(parts, ignore_index=True)
            assert job.smiles.str.len().gt(0).all(), f"empty SMILES in {f.name}"
            job.to_csv(f, index=False, compression="gzip")
            print(f"exported {f.name}: {len(F)} fit, {len(job) - len(F)} to predict", flush=True)


def train(tasks: list[str], workers: int, threads: int) -> None:
    jobs = [JOBS / f"{t}__s{s}.csv.gz" for t in ORDER if t in tasks for s in range(5)]
    todo = [j for j in jobs if not j.with_name(j.name.replace(".csv.gz", f"__{MODEL}.csv.gz")).exists()]

    def run(job: Path) -> str:
        log = job.with_name(job.name.replace(".csv.gz", f"__{MODEL}.log"))
        t0 = time.time()
        with open(log, "w") as fh:
            r = subprocess.run([str(GRAPH_PY), str(ROOT / "scripts" / "graph_train.py"), str(job),
                                "--threads", str(threads)], stdout=fh, stderr=subprocess.STDOUT)
        return f"{job.name}: exit {r.returncode}, {time.time() - t0:.0f} s"

    print(f"{len(todo)} of {len(jobs)} jobs to run, {workers} at a time, {threads} threads each", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for msg in pool.map(run, todo):
            print(msg, flush=True)


def merge(tasks: list[str]) -> None:
    for task in tasks:
        for seed, spec in TASKS[task]():
            out_f = PRED_M3 / f"{task}__{MODEL}__s{seed}.csv.gz"
            if out_f.exists():
                continue
            g = JOBS / f"{task}__s{seed}__{MODEL}.csv.gz"
            if not g.exists():
                print(f"missing {g.name}", flush=True)
                continue
            G = pd.read_csv(g)
            B, y, F = spec["B"], spec["y"], spec["F"]
            blocks = [(name, B[idx], y[idx], idx) for name, idx in spec["sets"].items()]
            for name, (edf, _, Be) in spec["ext"].items():
                blocks.append((name, Be, edf.y.values, np.arange(len(edf))))
            rows = []
            for name, Bs, ys, idx in blocks:
                gg = G[G.set == name].set_index("idx").loc[idx]
                assert np.allclose(gg.y.values, ys), f"label mismatch in {name}"
                s1, s5 = features.tanimoto_topk(Bs, B[F], k=5)
                part = pd.DataFrame({"set": name, "idx": idx, "y": ys, "yhat": gg.yhat.values,
                                     "u": gg.u.values, "s1": s1, "s5": s5})
                if name == "E2":
                    edf = spec["ext"]["E2"][0]
                    part["sp3_CF"] = edf.sp3_CF.values
                    part["f_group"] = edf.f_group.values
                if "ligand" in spec["meta"] and name not in spec["ext"]:
                    part["group"] = spec["meta"]["ligand"][idx]
                rows.append(part)
            P = pd.concat(rows)
            ref = pd.read_csv(PRED / f"{task}__RF__s{seed}.csv.gz")
            assert (ref.set.values == P.set.values).all() and (ref.idx.values == P.idx.values).all(), "row order differs from M1"
            assert np.allclose(ref.s1.values, P.s1.values) and np.allclose(ref.s5.values, P.s5.values), "s1/s5 differ from M1"
            PRED_M3.mkdir(parents=True, exist_ok=True)
            P.to_csv(out_f, index=False, compression="gzip")
            meta = {k: v for k, v in spec["meta"].items() if k != "ligand"}
            gm = json.loads(g.with_suffix("").with_suffix(".json").read_text())
            meta.update({"n_fit": int(len(F)), **{f"n_{k}": int(len(v)) for k, v in spec["sets"].items()},
                         "seconds": gm["seconds"], "graph": gm})
            (PRED_M3 / f"{task}__{MODEL}__s{seed}.json").write_text(json.dumps(meta, indent=1, default=str))
            print(f"{task:10s} {MODEL} seed {seed}: merged {len(P)} rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["export", "train", "merge"])
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    if a.step == "export":
        export(tasks)
    elif a.step == "train":
        train(tasks, a.workers, a.threads)
    else:
        merge(tasks)


if __name__ == "__main__":
    sys.exit(main())
