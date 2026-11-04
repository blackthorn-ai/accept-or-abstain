# Provenance

Every input to this repository is listed here with its source and licence. No client,
proprietary or confidential data, code, model, result, report or presentation was
used. The commit history starts from an empty repository (commit `4d98d6b`), and the
analysis protocol (`PROTOCOL.md`) was committed before any test-set result was
computed (commit `f187a12`; clarifications in `0269f67`, also before scoring).

## Data

| File (in `data/raw/`, fetched by `scripts/fetch_data.py`) | Source | Licence | SHA-256 |
|---|---|---|---|
| `SangsterLogP_Datasets.xlsx` | Cirino, Caron, Ermondi, Charochkina, Tetko. SangsterLogP dataset, v1. Zenodo, doi:10.5281/zenodo.19387551 (record 19387552) | CC BY 4.0 | `a5f998a6…a8d42` |
| `Buchwald_Hartwig.xlsx` | Ahneman, Estrada, Lin, Dreher, Doyle, *Science* 2018, 360, 186; redistributed in `SoftServeInc/yield-paper` at commit `8085ecf` | MIT (redistribution) | `d7f318ab…2ee2e` |
| `Suzuki-Miyaura.xlsx` | Perera et al., *Science* 2018, 359, 429; redistributed in `SoftServeInc/yield-paper` at commit `8085ecf` | MIT (redistribution) | `a869e020…84640` |
| `fluoric_IDs_SMILES_importances_pKa_logP.csv` | Measured logP of fluorinated amides and their parents; `blackthorn-ai/fluoricLogPpKa` at commit `389be10` (data of the 2025 *J. Comput. Chem.* study, doi:10.1002/jcc.70002) | MIT | `35f93062…e909e8b` |

Full digests are in `scripts/fetch_data.py`.

The structures of the Suzuki–Miyaura components in `src/aoa/data.py` are the
published identities of the reagents (Perera et al. 2018). Spectator fragments
(internal standard, HCl, water of hydration) are omitted; the table was written for
this repository.

## Code

All code in `src/` and `scripts/` was written for this repository. It depends only on
published open-source libraries:

| Library | Use | Licence |
|---|---|---|
| RDKit | parsing, ECFP4, descriptors, InChIKey | BSD-3-Clause |
| scikit-learn | random forest, MLP, logistic regression | BSD-3-Clause |
| NumPy, SciPy, pandas | numerics, binomial tail, tables | BSD-3-Clause |
| matplotlib | figures | PSF-based (matplotlib licence) |
| openpyxl | reading .xlsx | MIT |

No code from the `SoftServeInc` repositories is used; only their data files.
