import unittest

import numpy as np
import pandas as pd

from aoa.reporting import gate_reports


def example():
    rows = []
    for seed, n, k in [(0, 1, 1), (1, 9, 0), (2, 0, 0)]:
        key = dict(task="example", model="RF", seed=seed, cal="C_h", test="H")
        rows.append(dict(**key, gate="none", n=1000, n_acc=1000, target=np.nan,
                         coverage=1.0, precision=0.5))
        rows.append(dict(**key, gate="LTT_fused", target=0.8, n_acc=n,
                         coverage=n / 1000, precision=k/n if n else np.nan))
    return pd.DataFrame(rows)


class ReportingTests(unittest.TestCase):
    def report(self, d):
        return gate_reports(d, [("D1", "example", "H")])

    def test_weighting_counts_and_undefined_precision(self):
        seeds, pooled = self.report(example())
        cell = pooled.iloc[0]
        self.assertEqual((cell.n_correct, cell.n_acc, cell.seeds_accepting), (1, 10, 2))
        self.assertAlmostEqual(cell.pooled_precision, 0.1)
        self.assertAlmostEqual(cell.mean_precision, 0.5)
        self.assertAlmostEqual(cell.coverage, 10 / 3000)
        self.assertTrue(np.isnan(seeds[seeds.seed == 2].precision.iloc[0]))
        self.assertEqual(seeds[seeds.seed == 0].n_acc.iloc[0], 1)

    def test_all_abstaining_stays_undefined(self):
        d = example()
        gate = d.gate == "LTT_fused"
        d.loc[gate, ["n_acc", "coverage"]] = 0
        d.loc[gate, "precision"] = np.nan
        _, pooled = self.report(d)
        self.assertTrue(pooled.pooled_precision.isna().all())
        self.assertTrue(pooled.mean_precision.isna().all())

    def test_inconsistent_counts_are_rejected(self):
        d = example()
        d.loc[(d.gate == "LTT_fused") & (d.seed == 0), "precision"] = 0.5
        with self.assertRaisesRegex(ValueError, "integer"):
            self.report(d)

    def test_missing_test_denominator_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "test counts"):
            self.report(example().query("gate != 'none'"))

    def test_duplicate_seed_rows_are_rejected(self):
        d = example()
        with self.assertRaises(ValueError):
            self.report(pd.concat([d, d[d.gate == "LTT_fused"]]))

    def test_fractional_counts_are_not_truncated(self):
        for column in ("n", "n_acc"):
            with self.subTest(column=column):
                d = example()
                first_gate = (d.gate == "LTT_fused") & (d.seed == 0)
                if column == "n":
                    d.loc[(d.gate == "none") & (d.seed == 0), "n"] = 1000.5
                    d.loc[first_gate, "coverage"] = 1 / 1000.5
                else:
                    d["n_acc"] = d.n_acc.astype(float)
                    d.loc[first_gate, "n_acc"] = 1.5
                    d.loc[first_gate, "precision"] = 2 / 3
                    d.loc[first_gate, "coverage"] = 1.5 / 1000
                with self.assertRaisesRegex(ValueError, "finite integers"):
                    self.report(d)

    def test_nonfinite_counts_are_rejected(self):
        d = example()
        first_gate = (d.gate == "LTT_fused") & (d.seed == 0)
        d.loc[(d.gate == "none") & (d.seed == 0), "n"] = np.inf
        d.loc[first_gate, "coverage"] = 0
        with self.assertRaisesRegex(ValueError, "finite integers"):
            self.report(d)


if __name__ == "__main__":
    unittest.main()
