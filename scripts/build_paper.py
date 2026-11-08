"""Build the current working manuscript, SI and synchronized source package.

Requires pdflatex, bibtex and xelatex on PATH. Does not submit or publish anything.
The posted chemrxiv.15011706_v1.pdf is never replaced.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def main():
    logs = []
    for args in [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "si.tex"],
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "si.tex"],
    ]:
        run = subprocess.run(args, cwd=PAPER, text=True, capture_output=True)
        logs.append(" ".join(args) + "\n" + run.stdout + run.stderr)
        (PAPER / "build.log").write_text("\n".join(logs))
        if run.returncode:
            raise RuntimeError(f"{args[0]} failed; see paper/build.log")
    shutil.copyfile(PAPER / "main.pdf", ROOT / "main.pdf")
    paths = [*PAPER.glob("*.tex"), *PAPER.glob("*.bib"),
             PAPER / "main.pdf", PAPER / "si.pdf",
             *PAPER.glob("figures/*.pdf"), *PAPER.glob("figures/*.png"),
             *PAPER.glob("tables/*.tex")]
    with ZipFile(ROOT / "paper.zip", "w", ZIP_DEFLATED) as package:
        for path in sorted(paths):
            package.write(path, path.relative_to(ROOT))
        package.write(ROOT / "REVISION_NOTES.md", "REVISION_NOTES.md")
    print("Built paper/main.pdf, paper/si.pdf; synchronized main.pdf and paper.zip")


if __name__ == "__main__":
    main()
