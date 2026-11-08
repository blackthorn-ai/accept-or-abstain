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


def package_paths(root: Path) -> dict[str, Path]:
    """Required manuscript files plus figure/table assets, keyed by archive name."""
    paper = root / "paper"
    required = [paper / name for name in
                ("main.tex", "si.tex", "references.bib", "main.pdf", "si.pdf")]
    required.append(root / "REVISION_NOTES.md")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required source-package member is missing: {path}")
    paths = [*required, *paper.glob("*.tex"), *paper.glob("*.bib"),
             *paper.glob("figures/*.pdf"), *paper.glob("figures/*.png"),
             *paper.glob("tables/*.tex")]
    return {str(path.relative_to(root)): path for path in sorted(set(paths))}


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
    with ZipFile(ROOT / "paper.zip", "w", ZIP_DEFLATED) as package:
        for name, path in package_paths(ROOT).items():
            package.write(path, name)
    print("Built paper/main.pdf, paper/si.pdf; synchronized main.pdf and paper.zip")


if __name__ == "__main__":
    main()
