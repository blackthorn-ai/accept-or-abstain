"""Model M3 (protocol amendment 2.0): a five-member Chemprop D-MPNN ensemble.

Runs in the graph environment (.venv-graph; versions in env/graph-requirements.txt), not in the
main one, and imports nothing from `aoa`. Each member is trained on F only: a random 10% of F
selects the checkpoint (lowest validation loss over 50 epochs) and the other 90% trains it.
Settings are Chemprop's defaults and are not tuned. Member m of seed s uses the random seed
100 * s + m for its initialisation, data order and validation subset.

Usage: .venv-graph/bin/python scripts/graph_train.py results/graph_jobs/<task>__s<seed>.csv.gz [--threads 3]

Writes, next to the job file, <job>__DMPNN.csv.gz (set, idx, y, yhat = member mean,
u = member standard deviation) and <job>__DMPNN.json (timings, best epochs, versions).
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

N_MEMBERS = 5
EPOCHS = 50
BATCH = 64
VAL_FRAC = 0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", type=Path)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))

    import torch
    torch.set_num_threads(a.threads)
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    import chemprop
    import rdkit
    from chemprop import data, featurizers, models, nn

    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*predict_dataloader.*")
    job = pd.read_csv(a.job)
    seed = int(a.job.name.split("__s")[1].split(".")[0])
    fit = job[job.set == "F"].reset_index(drop=True)
    rest = job[job.set != "F"].reset_index(drop=True)
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def dataset(smiles, y=None):
        if y is None:
            dps = [data.MoleculeDatapoint.from_smi(s) for s in smiles]
        else:
            dps = [data.MoleculeDatapoint.from_smi(s, np.array([v], dtype=float)) for s, v in zip(smiles, y)]
        ds = data.MoleculeDataset(dps, feat)
        ds.cache = True
        return ds

    pred_ds = dataset(rest.smiles.tolist())
    pred_dl = data.build_dataloader(pred_ds, batch_size=512, num_workers=0, shuffle=False)

    def fit_member(m: int, ms: int, tmp: str):
        """Train one member with seed ms; return its predictions for every non-F row and a record."""
        tm = time.time()
        perm = np.random.default_rng(ms).permutation(len(fit))
        n_val = int(round(VAL_FRAC * len(fit)))
        va_i, tr_i = perm[:n_val], perm[n_val:]
        pl.seed_everything(ms, workers=True)
        tr = dataset(fit.smiles.values[tr_i], fit.y.values[tr_i])
        va = dataset(fit.smiles.values[va_i], fit.y.values[va_i])
        scaler = tr.normalize_targets()
        va.normalize_targets(scaler)
        ffn = nn.RegressionFFN(output_transform=nn.UnscaleTransform.from_standard_scaler(scaler))
        model = models.MPNN(nn.BondMessagePassing(), nn.NormAggregation(), ffn)
        tl = data.build_dataloader(tr, batch_size=BATCH, num_workers=0, seed=ms)
        vl = data.build_dataloader(va, batch_size=BATCH, num_workers=0, shuffle=False)
        ck = ModelCheckpoint(dirpath=os.path.join(tmp, f"seed{ms}"), filename="{epoch}", monitor="val_loss",
                             mode="min", save_top_k=1)
        trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="cpu", devices=1, logger=False,
                             enable_progress_bar=False, enable_model_summary=False, callbacks=[ck],
                             deterministic=True)
        trainer.fit(model, tl, vl)
        best = models.MPNN.load_from_checkpoint(ck.best_model_path)
        best.eval()
        runner = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
                            enable_model_summary=False)
        p = torch.cat(runner.predict(best, pred_dl)).numpy().reshape(-1)
        rec = {"member": m, "seed": ms, "n_train": int(len(tr_i)), "n_val": int(n_val),
               "best_val_loss": float(ck.best_model_score),
               "best_epoch": int(Path(ck.best_model_path).stem.split("epoch=")[-1]),
               "epochs_run": int(trainer.current_epoch), "finite": bool(np.isfinite(p).all()),
               "seconds": round(time.time() - tm, 1)}
        print(f"{a.job.name} member {m} (seed {ms}): best epoch {rec['best_epoch']}, "
              f"val loss {rec['best_val_loss']:.4f}, {rec['seconds']:.0f} s", flush=True)
        return p, rec

    preds, members = [], []
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for m in range(N_MEMBERS):
            # amendment 2.0, "Failures": one retry with seed + 50, then the member is dropped
            for ms in (100 * seed + m, 100 * seed + m + 50):
                p, rec = fit_member(m, ms, tmp)
                members.append(rec)
                if rec["finite"]:
                    preds.append(p)
                    break
    assert preds, "no member produced finite predictions"
    P = np.stack(preds)
    out = pd.DataFrame({"set": rest.set, "idx": rest.idx, "y": rest.y, "yhat": P.mean(0), "u": P.std(0)})
    stem = a.job.name.replace(".csv.gz", "")
    out.to_csv(a.job.with_name(f"{stem}__DMPNN.csv.gz"), index=False, compression="gzip")
    meta = {"job": a.job.name, "n_fit": int(len(fit)), "n_predicted": int(len(rest)),
            "n_members_used": len(preds), "members": members,
            "seconds": round(time.time() - t0, 1), "threads": a.threads,
            "versions": {"chemprop": chemprop.__version__, "torch": torch.__version__,
                         "lightning": pl.__version__, "rdkit": rdkit.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__}}
    a.job.with_name(f"{stem}__DMPNN.json").write_text(json.dumps(meta, indent=1))
    print(f"{a.job.name}: done in {meta['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
