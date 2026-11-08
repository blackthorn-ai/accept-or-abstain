# PR #1 follow-up review — 8 November 2026

Working assessment prepared with Codex at the author's request, initially against
`a919fc8e598a836b8f677e29830b7518ada86157`. Scope: scientific claim–implementation
alignment, repeated-prediction reporting, code edge cases, artifact verification,
reproduction instructions and merge readiness. This is not independent peer review,
a formal statistical proof or independent experimental/model validation. No new
base-model training or chemical measurements were performed.

## Findings and fixes

| Priority | Finding | Resolution |
|---|---|---|
| P2 | The revision verifier checked report counts but could miss altered precision/coverage values or incomplete source ZIPs. It trusted a local pickle cache; validation assertions disappeared under optimized Python. | Replay all 70 saved prediction files afresh, compare gate/score metrics and every per-seed/pooled value, check all expected ZIP members and bytes, and use explicit validation errors. Tamper and optimization regression tests added. |
| P2 | Fractional accepted/test counts could be silently truncated into misleading integer counts. | Require finite integer counts before conversion; regression tests include fractional and infinite counts. Valid experimental inputs and outputs are unchanged. |
| P2 | The unused empirical-cutoff helper could split tied scores when finding a cutoff, then deploy that cutoff to the whole tie, violating even its calibration-precision criterion. Scores [1,1], labels [1,0], target 0.8 returned a cutoff accepting both at precision 0.5. | Evaluate only complete tie blocks. Regression tests check this example and the loosest feasible tie block. The reported LTT-labelled and conjunctive gates do not call this helper. |
| P2 | E2 precision 0.67 was unlabeled beside a pooled card showing 0.71. The H1 caveat omitted partial overlap between seed-specific test sets. | Label E2 seed-averaged and pooled precision explicitly (24/34 pooled). State that original logP-H/SM-H bootstrap draws resample overlapping seed-specific sets separately. Positive H1 statements are identified as the original computational criterion. No interval or pass criterion was changed. |
| P3 | Instructions omitted graphical-abstract regeneration and Poppler, and verification required untracked TeX logs even when checking committed PDFs. | Include --ga and verification commands; document Poppler and allow clean-checkout PDF verification, with --require-build-logs for checks after a local rebuild. Package membership is shared between builder and verifier. |

The original saved RF held-out sets confirm partial overlap: logP-H seed pairs
share 312–824 query indices, and three SM-H seed pairs share 480 query indices.
The H1 bootstrap keeps common draws only when the complete index arrays match;
otherwise it resamples each seed separately. This is why its original intervals
remain historical diagnostics, not inference across new groups or training runs.

The manuscript also qualifies the cited LTT result by its sampling/testing
assumptions and describes the cited recalibration study's improvement as empirical
prediction-set coverage for classifiers. Source checks used the primary
[Learn then Test paper](https://arxiv.org/html/2110.01052v5),
[Morger et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9068909/), and
[NumPy quantile documentation](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html).

## Validation and retained limits

The regression suite includes adverse cases for tied cutoffs, fractional counts,
undefined precision, corrupted report values, missing/duplicate rows, mismatched
acceptance masks, omitted/stale archive members and optimized Python. Fresh replay
and document/package verification are recorded in
`results/revision/verification.json`. The 150 original baseline files, including
all original numerical result files and the posted v1 PDF, must retain their
SHA-256 digests. The four primary-model verdicts remain false; MLP H1 meets its
original criterion and H2 is true only vacuously. No new inferential analysis is
substituted for the withdrawn pooled intervals.

Remaining scientific limits are explicit in the paper: no proved finite-sample
gate/interval guarantee for these implementations, correlated query/group/seed
observations, non-isolated scorer/threshold calibration comparison, shared E1/E2
compounds and BH campaign dependence, incomplete H1 bootstrap pairing, and no
prospective experimental validation. These limits constrain interpretation; the
revision makes no claims that require them to have been resolved.

At initial inspection GitHub showed no reviews, unresolved threads, configured
check runs or workflow runs. Main was unprotected and no repository rulesets were
listed. Merge readiness is checked again on the final pushed head; merging this
code revision does not post a new ChemRxiv version.
