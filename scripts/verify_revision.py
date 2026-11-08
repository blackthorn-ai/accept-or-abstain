"""Verify the revision after saved-prediction replay and PDF/source-package build."""
from __future__ import annotations

import hashlib
import json
import pickle
import platform
import re
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "results" / "revision"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tests = subprocess.check_output(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT, stderr=subprocess.STDOUT, text=True,
    )
    tests_passed = int(re.search(r"Ran (\d+) tests", tests).group(1))
    baseline = json.loads((REV / "baseline_manifest.json").read_text())
    changed = [p for p, h in baseline["files"].items() if digest(ROOT / p) != h]
    if changed:
        raise AssertionError(f"Original observations/numerical results changed: {changed}")
    per_seed = pd.read_csv(REV / "gate_per_seed.csv")
    pooled = pd.read_csv(REV / "gate_pooled_descriptive.csv")
    assert len(per_seed) == 320 and len(pooled) == 64
    assert (pooled.seeds == 5).all()
    with open(ROOT / "results/cache/arrays.pkl", "rb") as stream:
        arrays = pickle.load(stream)
    for row in per_seed.itertuples():
        d = arrays[(row.task, row.model, row.seed, row.cal, row.test, "fused")]
        acc = d["acc08" if row.target == 0.8 else "acc09"]
        assert row.n_test == len(acc)
        assert row.n_acc == int(acc.sum())
        assert row.n_correct == int(d["correct"][acc].sum())
    assert digest(ROOT / "main.pdf") == digest(ROOT / "paper/main.pdf")
    with ZipFile(ROOT / "paper.zip") as package:
        for name in package.namelist():
            assert package.read(name) == (ROOT / name).read_bytes(), name
    pdfs = {}
    for name in ("main", "si"):
        path = ROOT / "paper" / f"{name}.pdf"
        text = subprocess.check_output(["pdftotext", str(path), "-"], text=True)
        assert "Working revision:" in text and "not posted" in text
        assert not re.search(r"\b(?:Table|Section|Figure)\s+\?\?", text)
        info = subprocess.check_output(["pdfinfo", str(path)], text=True)
        pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
        log = (ROOT / "paper" / f"{name}.log").read_text()
        assert "undefined references" not in log
        assert not re.search(r"Citation .* undefined", log)
        assert "Missing character:" not in log
        pdfs[str(path.relative_to(ROOT))] = {"pages": pages, "sha256": digest(path)}
    result = {
        "revision_date": "2026-11-08", "status": "unposted working revision",
        "base_commit": baseline["base_commit"],
        "scope": "Full saved-prediction analysis replay; no base-model retraining or independent validation",
        "baseline_files_checked": len(baseline["files"]), "baseline_mismatches": changed,
        "prediction_files": len(list((ROOT / "results/predictions").glob("*.csv.gz"))),
        "per_seed_rows_checked_against_replayed_acceptance_masks": len(per_seed),
        "pooled_descriptive_rows": len(pooled), "reporting_tests_passed": tests_passed,
        "hypotheses": json.loads((ROOT / "results/hypotheses.json").read_text()),
        "uncertainty_change": "Pooled Wilson intervals withdrawn; no replacement population interval",
        "original_h1_query_bootstrap": "Preserved; incomplete chemical-group and training uncertainty",
        "pdfs": pdfs, "root_pdf_and_source_zip_synchronized": True,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "commands": [
            "python -m unittest discover -s tests -v",
            "python scripts/analyse.py", "python scripts/make_figures.py --ga",
            "python scripts/make_tables.py", "python scripts/exploratory.py",
            "python scripts/build_paper.py", "python scripts/verify_revision.py",
        ],
    }
    (REV / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
