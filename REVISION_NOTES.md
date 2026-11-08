# Working revision — 8 November 2026

Prepared against repository commit `00e1188`. This is an author-review draft,
not a posted ChemRxiv version 2. The DOI in the README continues to identify v1.
The original posted `chemrxiv.15011706_v1.pdf`, protocol text and saved predictions
are preserved; earlier manuscript sources, PDFs and source package remain in Git.

## Changes to claims and interpretation

- Replaced the asserted 90% finite-sample selective-precision guarantee with an
  empirical binomial-test sequence inspired by Learn-then-Test, consistently in
  the manuscript, Figure 1, README and code documentation. The implemented rule,
  threshold candidate order, tie handling, minimum-count rule and delta are unchanged.
- Narrowed the random/held-out comparison to **complete pipelines**. It refits the
  fused scorer and changes threshold data and sample sizes while retaining the
  same base model. No isolated threshold-distribution effect, or claim that this
  effect exceeds signal choice, is supported by this comparison. No new ablation
  was introduced or retroactively prespecified.
- Documented the calibration-derived interval normalizer and exact original NumPy
  quantile convention. Empirical coverage is reported without claiming ordinary
  split-conformal validity, shifted-test exchangeability or accepted-subset coverage.
  The original interval calculation is retained to reproduce the observations.
- Retained the 182-compound E1/E2 overlap and identified shared campaign dependence;
  test/family reporting does not treat the eight tests as independent replications.
- Corrected the inference from negative R²: squared-error performance below the
  test-mean reference does not imply no ranking ability. Identified the E2 mean
  reference as test-label-derived and descriptive. Fixed the hypotheses cross-reference.
- Clarified the author's research, code, drafting, analysis and interpretation role,
  Claude's review/refinement role, and Codex's assistance with this revision, without
  inventing historical model/version details or claiming independent validation.
  This follows ChemRxiv's tool-and-use disclosure approach:
  https://axial.acs.org/publishing/new-chemrxiv-policy-on-the-use-of-ai-tools

## Uncertainty reporting and added descriptive outputs

The original pooled Wilson 95% intervals have been **withdrawn**, because accepted
prediction–seed observations repeat molecules/reactions and are dependent. Pooled
point precision is retained as the ratio of total correct to total accepted
prediction–seed observations. It describes the observed mixture; it is not a
precision estimate for independent new molecules, chemical groups or campaigns.
Coverage remains the mean over all five seeds, including zero-acceptance seeds.
Both calibration pipelines now show correct/accepted counts and accepting seeds.
Rare nonzero acceptance is visible in percentage coverage and exact counts.

New descriptive tables S5–S8 and `results/revision/gate_per_seed.csv` report every
held-out test, model, seed, calibration pipeline and target, with family, test count,
accepted count, correct count, coverage and precision. Precision stays undefined
when accepted count is zero. `gate_pooled_descriptive.csv` records the corresponding
summaries. These reporting extensions are post hoc; they introduce no new estimates
of uncertainty or population-level inference.

A paired/clustered inferential extension needs a defined target population and
resampling design for repeated queries and chemical/reaction groups. It has not
been substituted mechanically here. The original H1 query-bootstrap calculations
are retained only for the original criterion and are now explicitly limited:
they condition on fitted pipelines, resample partly overlapping logP-H/SM-H
queries independently across seeds, and do not fully address chemical-group/training
uncertainty, and do not establish generalization across new campaigns. Statistical
review of a future design remains necessary before such inferential claims.

## Numerical results and verification

Saved-prediction replay was performed for both model families across all 70
prediction files, without retraining base models. The original metric, score,
summary and hypothesis-JSON outputs are checked against the baseline hashes in
`results/revision/baseline_manifest.json`. All four primary-model hypotheses remain
false; MLP H1 remains true, H2 true only vacuously, H3/H4 false. Original observations
and point estimates are unchanged. The withdrawn pooled intervals are the change
to uncertainty reporting; no replacement confidence intervals were calculated.

Regression tests cover unequal seed weights, undefined abstention precision,
missing denominators, duplicate rows, fractional/nonfinite counts, tied empirical
cutoffs, corrupted report values and incomplete source packages. Verification now
recomputes masks and metrics from all saved predictions rather than trusting the
local pickle cache, and compares every per-seed and pooled value. Validation remains
active under optimized Python. The rebuilt
manuscript/SI, figures and source ZIP are checked for synchronization. Commands,
versions and detailed verification are recorded in `results/revision/verification.json`.
The old protocol wording in SI S3 is identified as historical and superseded where
it implied a guarantee. This replay and document check are not independent model
validation or a full experimental reproduction.

## Follow-up review of PR #1

The remaining E2 precision ambiguity is resolved explicitly: 0.67 is the mean over
accepting seeds; 0.71 is pooled (24/34 accepted prediction–seed observations).
The source package verifier now requires every expected member and rejects stale
bytes. The unused `empirical_threshold` helper was corrected to include whole score
ties; it previously could return a cutoff whose calibration precision was below
its stated target. This helper is not used by the published experimental pipelines,
so the correction changes none of their observations or hypothesis decisions.
The original H1 bootstrap is identified as a computational criterion with incomplete
pairing for partly overlapping seed-specific query sets. See `PR_REVIEW.md` for
review scope, findings, validation and retained scientific limitations.

## Build and publication status

Install the lockfile with `uv sync --frozen --python 3.12`. Replay from
`scripts/analyse.py`, regenerate figures/tables, run `scripts/exploratory.py`, then
`python3 scripts/build_paper.py`. The latter rebuilds both PDFs, synchronizes the
root `main.pdf`, and repackages current sources, figures, tables and PDFs in
`paper.zip`. It leaves the posted v1 PDF untouched. It needs TeX; pandoc is needed
only to regenerate the unchanged historical plan from its original wording.
Poppler (pdftotext/pdfinfo) is required by the verification script. A clean checkout
can verify committed PDFs after prediction replay without TeX logs; use
`--require-build-logs` after a local PDF build.

No ChemRxiv submission, new DOI assignment, petition exhibit replacement, CV or
ORCID change is part of this revision. Those references should change only after a
revision is actually posted and its platform-assigned DOI/version/date are known.
