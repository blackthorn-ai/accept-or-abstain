"""Build the SI's copy of the analysis plan (paper/tables/protocol_si.tex) from PROTOCOL.md.

Takes sections 1-10 of the plan as written and adds short implementation notes.
Needs pandoc on the PATH.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTES = """
## Notes on implementation

- A gate that abstains on every query has no defined precision. Such seeds are left
  out of the seed average. A cell in which every seed abstains counts as meeting the
  target in H2.
- H4 cannot be checked on a test where Gate B accepts nothing; this counts as a
  failure of H4 for that test.
- The held-out tests for H1–H4 are the logP cluster test, E1, E2 (all 183 compounds),
  BH Test1–Test4 and the SM ligand test.
- Holding out four calibration additives leaves about 1,640 reactions in F for each BH
  test. Ungated BH accuracy is therefore not directly comparable with published
  results on the same tests.
- The logP held-out sets were formed by assigning whole clusters until 15% of the
  compounds was reached.
- The Suzuki–Miyaura component structures come from a table written for this study
  (`src/aoa/data.py`). The table omits spectator fragments: the internal standard, HCl
  and water of hydration.
- For test sets that are identical in every seed (E1, E2, BH Test1–Test4), bootstrap
  intervals use the same resample for all seeds.
"""


def main():
    text = (ROOT / "PROTOCOL.md").read_text()
    body = text[text.index("## 1. Question"): text.index("## Changes after this version")]
    md = body.rstrip() + "\n" + NOTES
    if "## Amendment 2.0" in text:  # the graph network M3, fixed before any M3 prediction
        md += "\n" + text[text.index("## Amendment 2.0"):].rstrip() + "\n"
    out = subprocess.run(["pandoc", "-f", "gfm", "-t", "latex"], input=md, capture_output=True,
                         text=True, check=True).stdout
    out = out.replace("τ₁", "$\\tau_1$").replace("τ₂", "$\\tau_2$")
    (ROOT / "paper" / "tables" / "protocol_si.tex").write_text(out)
    print("wrote paper/tables/protocol_si.tex")


if __name__ == "__main__":
    main()
