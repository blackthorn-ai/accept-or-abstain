"""Verify saved predictions, report values, committed PDFs and the source package.

No model retraining or data download is needed. --require-build-logs additionally
requires local TeX logs from a fresh scripts/build_paper.py run. Poppler's
pdftotext/pdfinfo executables are required for the PDF checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

import analyse
from aoa.reporting import gate_reports
from build_paper import package_paths

ROOT = Path(__file__).resolve().parents[1]
REPORT_KEYS = ["family", "task", "test", "model", "seed", "cal", "target"]


def require(condition, message):
    """Validation must also execute when Python is run with optimization enabled."""
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_table(actual, expected, keys, name):
    require(set(actual.columns) == set(expected.columns), f"{name}: columns differ")
    require(not actual.duplicated(keys).any(), f"{name}: duplicate keys")
    a = actual.sort_values(keys).reset_index(drop=True).sort_index(axis=1)
    e = expected.sort_values(keys).reset_index(drop=True).sort_index(axis=1)
    try:
        pd.testing.assert_frame_equal(a, e, check_dtype=False, check_exact=False,
                                      rtol=1e-12, atol=1e-12)
    except AssertionError as error:
        raise ValueError(f"{name}: values or row keys differ from recomputed results") from error


def replay_predictions(root):
    """Recompute gate masks and metrics from saved predictions, without loading pickle."""
    analyse.FULL.clear()
    rows, scores = [], []
    files = sorted((root / "results/predictions").glob("*.csv.gz"))
    require(bool(files), "No saved prediction files")
    for path in files:
        r, s = analyse.analyse_file(path, {})
        rows.extend(r)
        scores.extend(s)
    keys = ["task", "model", "seed", "cal", "test"]
    compare_table(pd.read_csv(root / "results/metrics_long.csv"), pd.DataFrame(rows),
                  keys + ["gate", "target"], "Gate metrics")
    compare_table(pd.read_csv(root / "results/scores_long.csv"), pd.DataFrame(scores),
                  keys + ["score"], "Score metrics")
    return pd.DataFrame(rows), analyse.FULL, len(files)


def validate_gate_reports(per_seed, pooled, long, arrays, held_out=None):
    expected_seed, expected_pool = gate_reports(long, held_out or analyse.HELD_OUT)
    compare_table(per_seed, expected_seed, REPORT_KEYS, "Per-seed report")
    compare_table(pooled, expected_pool, [k for k in REPORT_KEYS if k != "seed"], "Pooled report")
    for row in per_seed.itertuples():
        d = arrays[(row.task, row.model, row.seed, row.cal, row.test, "fused")]
        require(row.target in (0.8, 0.9), "Unexpected gate target")
        acc = d["acc08" if row.target == 0.8 else "acc09"]
        n, k = int(acc.sum()), int(d["correct"][acc].sum())
        require(row.n_test == len(acc) and row.n_acc == n and row.n_correct == k,
                "Per-seed counts disagree with fresh acceptance masks")
        require(np.isclose(row.coverage, n / len(acc), rtol=0, atol=1e-12),
                "Per-seed coverage disagrees with fresh acceptance masks")
        require(np.isnan(row.precision) if n == 0 else
                np.isclose(row.precision, k / n, rtol=0, atol=1e-12),
                "Per-seed precision disagrees with fresh acceptance masks")


def verify_source_package(root):
    expected = package_paths(root)
    with ZipFile(root / "paper.zip") as package:
        names = package.namelist()
        require(len(names) == len(set(names)), "Source ZIP contains duplicate members")
        require(set(names) == set(expected), "Source ZIP has missing or unexpected members")
        for name, path in expected.items():
            require(package.read(name) == path.read_bytes(), f"Source ZIP member differs: {name}")


def verify_pdfs(root, require_build_logs=False):
    require(digest(root / "main.pdf") == digest(root / "paper/main.pdf"),
            "Root main.pdf differs from paper/main.pdf")
    pdfs = {}
    for name in ("main", "si"):
        path = root / "paper" / f"{name}.pdf"
        text = subprocess.check_output(["pdftotext", str(path), "-"], text=True)
        require("Working revision:" in text and "not posted" in text, "Missing revision status")
        require("??" not in text, "Unresolved PDF cross-reference")
        info = subprocess.check_output(["pdfinfo", str(path)], text=True)
        pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
        log_path = path.with_suffix(".log")
        require(not require_build_logs or log_path.is_file(), f"Build log missing: {log_path.name}")
        if log_path.is_file():
            log = log_path.read_text()
            require("undefined references" not in log and
                    not re.search(r"Citation .* undefined", log), "Unresolved TeX reference/citation")
            require("Missing character:" not in log, "Missing glyph in TeX output")
            require("Overfull \\hbox" not in log, "Overfull text in TeX output")
        pdfs[str(path.relative_to(root))] = {
            "pages": pages, "sha256": digest(path), "build_log_checked": log_path.is_file(),
        }
    return pdfs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-build-logs", action="store_true")
    args = parser.parse_args()
    rev = ROOT / "results/revision"
    tests = subprocess.check_output(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT, stderr=subprocess.STDOUT, text=True,
    )
    tests_passed = int(re.search(r"Ran (\d+) tests", tests).group(1))
    require(tests_passed > 0, "No regression tests were discovered")
    baseline = json.loads((rev / "baseline_manifest.json").read_text())
    changed = [p for p, h in baseline["files"].items() if digest(ROOT / p) != h]
    require(not changed, f"Original observations/numerical results changed: {changed}")
    long, arrays, files = replay_predictions(ROOT)
    per_seed = pd.read_csv(rev / "gate_per_seed.csv")
    pooled = pd.read_csv(rev / "gate_pooled_descriptive.csv")
    validate_gate_reports(per_seed, pooled, long, arrays)
    verify_source_package(ROOT)
    pdfs = verify_pdfs(ROOT, args.require_build_logs)
    result = {
        "revision_date": "2026-11-08", "status": "unposted working revision",
        "base_commit": baseline["base_commit"],
        "scope": "Fresh saved-prediction metric replay; no base-model retraining or independent validation",
        "baseline_files_checked": len(baseline["files"]), "baseline_mismatches": changed,
        "prediction_files": files, "gate_metric_rows_replayed": len(long),
        "per_seed_rows_checked_against_replayed_acceptance_masks": len(per_seed),
        "per_seed_and_pooled_values_recomputed": True, "pooled_descriptive_rows": len(pooled),
        "reporting_tests_passed": tests_passed,
        "hypotheses": json.loads((ROOT / "results/hypotheses.json").read_text()),
        "uncertainty_change": "Pooled Wilson intervals withdrawn; no replacement population interval",
        "original_h1_query_bootstrap": "Preserved; incomplete partial-overlap, chemical-group and training uncertainty",
        "pdfs": pdfs, "root_pdf_and_source_zip_synchronized": True,
        "source_zip_membership_checked": True,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "commands": [
            "python -m unittest discover -s tests -v",
            "python scripts/analyse.py", "python scripts/make_figures.py --ga",
            "python scripts/make_tables.py", "python scripts/exploratory.py",
            "python scripts/build_paper.py", "python scripts/verify_revision.py --require-build-logs",
        ],
    }
    (rev / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
