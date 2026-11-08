import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_paper import package_paths
from test_reporting import example
from verify_revision import require, validate_gate_reports, verify_pdfs, verify_source_package
from aoa.reporting import gate_reports

HELD_OUT = [("D1", "example", "H")]


def report_fixture():
    long = example()
    per_seed, pooled = gate_reports(long, HELD_OUT)
    arrays = {}
    for seed, n, k in [(0, 1, 1), (1, 9, 0), (2, 0, 0)]:
        acc = np.arange(1000) < n
        correct = np.arange(1000) < k
        arrays[("example", "RF", seed, "C_h", "H", "fused")] = {
            "acc08": acc, "correct": correct,
        }
    return per_seed, pooled, long, arrays


class VerificationTests(unittest.TestCase):
    def test_consistent_reports_pass(self):
        validate_gate_reports(*report_fixture(), held_out=HELD_OUT)

    def test_corrupt_per_seed_values_are_rejected(self):
        for column in ("precision", "coverage"):
            with self.subTest(column=column):
                per_seed, pooled, long, arrays = report_fixture()
                per_seed.loc[0, column] = 0.25
                with self.assertRaisesRegex(ValueError, "Per-seed report"):
                    validate_gate_reports(per_seed, pooled, long, arrays, HELD_OUT)

    def test_corrupt_pooled_values_are_rejected(self):
        for column in ("pooled_precision", "mean_precision", "coverage", "n_acc",
                       "n_correct", "seeds", "seeds_accepting", "n_prediction_seed"):
            with self.subTest(column=column):
                per_seed, pooled, long, arrays = report_fixture()
                pooled[column] = pooled[column].astype(float)
                pooled.loc[0, column] = 0.25
                with self.assertRaisesRegex(ValueError, "Pooled report"):
                    validate_gate_reports(per_seed, pooled, long, arrays, HELD_OUT)

    def test_missing_or_duplicate_report_rows_are_rejected(self):
        import pandas as pd
        for change in ("missing", "duplicate"):
            with self.subTest(change=change):
                per_seed, pooled, long, arrays = report_fixture()
                per_seed = per_seed.iloc[1:] if change == "missing" else pd.concat([per_seed, per_seed.iloc[:1]])
                with self.assertRaises(ValueError):
                    validate_gate_reports(per_seed, pooled, long, arrays, HELD_OUT)

    def test_report_counts_are_checked_against_fresh_masks(self):
        per_seed, pooled, long, arrays = report_fixture()
        arrays[("example", "RF", 0, "C_h", "H", "fused")]["correct"][:] = False
        with self.assertRaisesRegex(ValueError, "fresh acceptance masks"):
            validate_gate_reports(per_seed, pooled, long, arrays, HELD_OUT)

    def test_validation_stays_active_with_python_optimization(self):
        with self.assertRaisesRegex(ValueError, "bad"):
            require(False, "bad")
        script = "import sys; sys.path.insert(0, 'scripts'); from verify_revision import require; require(False, 'optimized check')"
        result = subprocess.run([sys.executable, "-O", "-c", script],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("optimized check", result.stderr)

    def test_source_zip_requires_all_members_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "paper").mkdir()
            for name in ("main.tex", "si.tex", "references.bib", "main.pdf", "si.pdf"):
                (root / "paper" / name).write_text(name)
            (root / "REVISION_NOTES.md").write_text("revision")
            for change in ("none", "missing", "changed"):
                with self.subTest(change=change):
                    with ZipFile(root / "paper.zip", "w") as package:
                        for name, path in package_paths(root).items():
                            if change == "missing" and name == "paper/main.tex":
                                continue
                            content = b"stale" if change == "changed" and name == "paper/main.pdf" else path.read_bytes()
                            package.writestr(name, content)
                    if change == "none":
                        verify_source_package(root)
                    else:
                        with self.assertRaises(ValueError):
                            verify_source_package(root)

    def test_committed_pdf_checks_do_not_require_untracked_build_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "paper").mkdir()
            for name in ("main.pdf", "paper/main.pdf", "paper/si.pdf"):
                (root / name).write_bytes(b"example PDF bytes")
            text = "Working revision: 8 November 2026 (not posted)"
            with patch("verify_revision.subprocess.check_output",
                       side_effect=[text, "Pages: 1", text, "Pages: 1"]):
                pdfs = verify_pdfs(root)
            self.assertTrue(all(not item["build_log_checked"] for item in pdfs.values()))
            with patch("verify_revision.subprocess.check_output", side_effect=[text, "Pages: 1"]):
                with self.assertRaisesRegex(ValueError, "Build log missing"):
                    verify_pdfs(root, require_build_logs=True)


if __name__ == "__main__":
    unittest.main()
