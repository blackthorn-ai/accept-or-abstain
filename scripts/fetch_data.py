"""Download the public datasets at pinned versions and check their SHA-256 digests."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
YIELD = "https://raw.githubusercontent.com/SoftServeInc/yield-paper/8085ecfbfc32523ed434c4c93020b5a56bc1f626/data"
FLUORIC = "https://raw.githubusercontent.com/blackthorn-ai/fluoricLogPpKa/389be10a3db40ad68051dafe8f186d6cc5a08ca4/fluoriclogppka/data"
FILES = {
    "SangsterLogP_Datasets.xlsx": (
        "https://zenodo.org/api/records/19387552/files/Datasets.xlsx/content",
        "a5f998a690ff48d820afe11b23d2522e95ff11764f6d13c773749750f3ea8d42"),
    "Buchwald_Hartwig.xlsx": (
        f"{YIELD}/Buchwald_Hartwig.xlsx",
        "d7f318abfbb42eaadca482b1e9aa54540eeed60e0089b6a0eb4e75366c31ee2e"),
    "Suzuki-Miyaura.xlsx": (
        f"{YIELD}/Suzuki-Miyaura.xlsx",
        "a869e020ba31bd5676c67a4791c3b7384711b5216de6af444b8cd0a24c284640"),
    "fluoric_IDs_SMILES_importances_pKa_logP.csv": (
        f"{FLUORIC}/IDs_SMILES_importances_pKa_logP.csv",
        "35f93062a28aee1e040125331bd87eccf40427ce65054fbd87e30549ca909e8b"),
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in FILES.items():
        f = RAW / name
        if not f.exists():
            print("downloading", name)
            urllib.request.urlretrieve(url, f)
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != digest:
            raise SystemExit(f"{name}: SHA-256 {got} does not match {digest}")
        print("ok", name)


if __name__ == "__main__":
    main()
