# accept-or-abstain

A held-out-calibrated accept-or-abstain protocol for molecular property and
reaction-yield models, and the code for the preprint *Accept or abstain? A
pre-specified test of reliability gates calibrated on held-out chemistry for molecular
property and reaction-yield models* (O. Gurbych, ChemRxiv, 2026,
[doi:10.26434/chemrxiv.15011706/v1](https://doi.org/10.26434/chemrxiv.15011706/v1);
a preprint, not peer reviewed).

The model returns a prediction only when a gate accepts the query. The gate scores
each query by its Tanimoto similarity to the training set and by ensemble spread. Its
threshold is selected by an empirical binomial-test sequence inspired by
Learn-then-Test on a calibration fold held out *by chemistry*. No finite-sample
selective-precision guarantee is established for this implementation. The hypotheses and success criteria are in
[`PROTOCOL.md`](PROTOCOL.md).

## Result in one paragraph

For the primary random forest, on seven of eight held-out tests, ranking queries by the gate's score lowered the
error rate at 30% coverage by 0.16–0.35. At a target precision of 0.8, the complete random-fold and held-out-fold
pipelines differed most for reaction yields. Switching folds refits the fused
scorer and changes threshold data and their sizes; it does not isolate threshold
calibration. Random-fold gates accepted 94–99% of Buchwald–Hartwig reactions with unseen
additives, at a precision of 0.59–0.75. Held-out-fold gates reached a pooled
precision of 0.85–0.87 on three of the four tests, at 29–62% coverage. At a target of
0.9 the difference was smaller. For the random forest, fusing similarity with spread
added nothing to spread alone, and conformal intervals calibrated on a random fold
lost coverage under shift. **All four pre-specified hypotheses failed their complete
criteria for the primary model.** In an exploratory analysis, intervals calibrated on the held-out fold
increased empirical coverage, at the cost of width; no coverage guarantee under
shift or after gate selection is established. See [`results/hypotheses.md`](results/hypotheses.md),
`results/exploratory_RF.json` and protocol note 1.2 in [`PROTOCOL.md`](PROTOCOL.md).

## This revision (8 November 2026)

This branch contains an **unposted working revision**. The DOI below identifies the
posted version 1. The original posted PDF, original protocol, raw saved predictions
and their Git history are preserved. See [REVISION_NOTES.md](REVISION_NOTES.md).

Operating cards report **descriptive** precision pooled over accepted
prediction–seed observations, exact correct/accepted counts, and accepting seeds
for both pipelines. Repeated predictions are correlated; pooled Wilson intervals
have been removed. [Per-seed results](results/revision/gate_per_seed.csv) and
[descriptive pooled summaries](results/revision/gate_pooled_descriptive.csv) are
provided by dataset/test family. No new-group or new-campaign precision interval
is claimed. The original H1 query bootstrap does not fully preserve dependence from partly
overlapping seed-specific query sets or chemical groups, and omits training-set
uncertainty. Its criterion is retained as a historical computational result. E1/E2 share 182 compounds, and BH tests share a campaign;
the eight tests are not eight independent replications.

The interval implementation retains its calibration-derived normalizer and original
NumPy quantile convention for reproducibility. Ordinary split-conformal validity
has not been established for these choices; chemistry-held-out folds need not be
exchangeable with their tests, and gate selection does not preserve marginal
coverage automatically. The historical API name `ltt_threshold` and `LTT_*` result
labels refer to the empirical rule, not a proved guarantee.

## Reproduce

```bash
uv sync --frozen --python 3.12
.venv/bin/python scripts/fetch_data.py        # public data, SHA-256 checked
.venv/bin/python scripts/run_predictions.py   # base models, per-query predictions
# M3, the graph network of protocol amendment 2.0 (separate environment: Chemprop needs an older RDKit)
uv venv .venv-graph --python 3.12 && uv pip install --python .venv-graph/bin/python -r env/graph-requirements.txt
PYTHONPATH=scripts:src .venv/bin/python scripts/graph_jobs.py export
PYTHONPATH=scripts:src .venv/bin/python scripts/graph_jobs.py train --workers 6 --threads 3
PYTHONPATH=scripts:src .venv/bin/python scripts/graph_jobs.py merge
.venv/bin/python scripts/analyse.py           # gates, metrics, hypotheses
.venv/bin/python scripts/make_figures.py --ga
.venv/bin/python scripts/make_tables.py
.venv/bin/python scripts/exploratory.py      # exploratory analyses (not in the plan)
.venv/bin/python scripts/make_si_plan.py     # SI copy of the plan (needs pandoc)
python3 scripts/build_paper.py
.venv/bin/python scripts/verify_revision.py --require-build-logs
```

To replay the saved predictions, start at `scripts/analyse.py`; no base-model
retraining or data download is required. Then run the figure, table and exploratory
scripts and build the PDFs. The build requires TeX (pdflatex, bibtex and xelatex);
regenerating the unchanged historical plan additionally requires pandoc. Verification
also needs Poppler's `pdftotext` and `pdfinfo`. To verify committed PDFs in a clean
checkout after replaying the predictions, use `scripts/verify_revision.py` without
`--require-build-logs`; the flag additionally requires logs from a local TeX build.

The full run takes about 40 minutes on an 18-core laptop, most of it the MLP ensembles.

## Layout

| Path | Contents |
|---|---|
| `PROTOCOL.md` | the analysis plan, with dated notes |
| `PROVENANCE.md` | every dataset and library, with licence and digest |
| `src/aoa/` | data loaders, features, splits, models, empirical gates and conformal-style interval rules, metrics |
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
