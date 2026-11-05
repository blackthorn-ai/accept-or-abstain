# accept-or-abstain

A held-out-calibrated accept-or-abstain protocol for molecular property and
reaction-yield models, and the code for the preprint *Accept or abstain? A
pre-specified test of reliability gates calibrated on held-out chemistry for molecular
property and reaction-yield models* (O. Gurbych, ChemRxiv, 2026,
[doi:10.26434/chemrxiv.15011706/v1](https://doi.org/10.26434/chemrxiv.15011706/v1);
a preprint, not peer reviewed).

The model returns a prediction only when a gate accepts the query. The gate scores
each query by its Tanimoto similarity to the training set and by ensemble spread. Its
threshold is set by Learn-then-Test to a target precision on a calibration fold held
out *by chemistry*. The hypotheses and success criteria are in
[`PROTOCOL.md`](PROTOCOL.md).

## Result in one paragraph

On seven of eight held-out tests, ranking queries by the gate's score lowered the
error rate at 30% coverage by 0.16–0.35. At a target precision of 0.8, calibrating
the threshold on a held-out fold instead of a random one mattered most for reaction
yields. Random-fold gates accepted 94–99% of Buchwald–Hartwig reactions with unseen
additives, at a precision of 0.59–0.75. Held-out-fold gates reached a pooled
precision of 0.85–0.87 on three of the four tests, at 29–62% coverage. At a target of
0.9 the difference was smaller. For the random forest, fusing similarity with spread
added nothing to spread alone, and conformal intervals calibrated on a random fold
lost coverage under shift. **All four pre-specified hypotheses failed their
criteria.** In an exploratory analysis, intervals calibrated on the held-out fold
recovered coverage, at the cost of width. See [`results/hypotheses.md`](results/hypotheses.md),
`results/exploratory_RF.json` and protocol note 1.2 in [`PROTOCOL.md`](PROTOCOL.md).

## Reproduce

```bash
uv venv --python 3.12 && uv pip install -e .
.venv/bin/python scripts/fetch_data.py        # public data, SHA-256 checked
.venv/bin/python scripts/run_predictions.py   # base models, per-query predictions
.venv/bin/python scripts/analyse.py           # gates, metrics, hypotheses
.venv/bin/python scripts/make_figures.py
.venv/bin/python scripts/make_tables.py
.venv/bin/python scripts/exploratory.py      # exploratory analyses (not in the plan)
.venv/bin/python scripts/make_si_plan.py     # SI copy of the plan (needs pandoc)
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main && xelatex si
```

The full run takes about 40 minutes on an 18-core laptop, most of it the MLP ensembles.

## Layout

| Path | Contents |
|---|---|
| `PROTOCOL.md` | the analysis plan, with dated notes |
| `PROVENANCE.md` | every dataset and library, with licence and digest |
| `src/aoa/` | data loaders, features, splits, models, gates (Learn-then-Test, conformal), metrics |
| `scripts/` | the pipeline, in the order above |
| `results/predictions/` | one file per task, model and seed: every query's prediction, spread and similarity |
| `results/` | metric tables, `hypotheses.md` |
| `paper/` | LaTeX source, figures and tables of the preprint |

## How to cite

Gurbych, O. Accept or abstain? A pre-specified test of reliability gates calibrated on
held-out chemistry for molecular property and reaction-yield models. *ChemRxiv* 2026.
https://doi.org/10.26434/chemrxiv.15011706/v1 (preprint, not peer reviewed).

```bibtex
@article{gurbych2026acceptorabstain,
  author  = {Gurbych, Oleksandr},
  title   = {Accept or abstain? {A} pre-specified test of reliability gates calibrated
             on held-out chemistry for molecular property and reaction-yield models},
  journal = {ChemRxiv},
  year    = {2026},
  doi     = {10.26434/chemrxiv.15011706/v1},
  url     = {https://doi.org/10.26434/chemrxiv.15011706/v1},
  note    = {Preprint, not peer reviewed}
}
```

## Licence

Code: MIT. Data: see `PROVENANCE.md` (CC BY 4.0 and MIT).
