# Analysis protocol (fixed before any test-set result)

Version 1.0, 4 November 2026. This file is committed to the repository before any
test-set prediction is scored. Any later change is recorded at the end of this file
with its date and reason; the original text is not edited.

## 1. Question

Can an accept-or-abstain gate, calibrated on data held out *by chemistry*, keep the
precision of a molecular model's accepted predictions at a stated target when the
queries come from chemistry the model has not seen? And does combining an
input-distance signal with a model-confidence signal rank errors better than either
signal alone?

## 2. Definitions fixed in advance

- **Correct prediction.** LogP: |y − ŷ| ≤ 0.5 log units. Reaction yield: |y − ŷ| ≤ 10
  percentage points.
- **Coverage** is the share of queries the gate accepts. **Selective precision** is the
  share of accepted queries that are correct. **Selective error rate** = 1 − selective
  precision.
- **Target precisions:** t = 0.8 and t = 0.9. Risk level for the threshold guarantee:
  δ = 0.1.

## 3. Data (all public)

- **D1 LogP pool:** SangsterLogP main sheet (Zenodo 10.5281/zenodo.19387551, CC BY
  4.0). Parsed with RDKit; unparsable SMILES dropped; duplicates by InChIKey merged
  (first record kept). Every compound whose InChIKey occurs in either external set
  below is removed from the pool.
- **E1, prospective external set:** the SangsterLogP "Prospective" sheet (327 compounds).
- **E2, fluorinated external set:** the 183 measured amide logP values of
  `blackthorn-ai/fluoricLogPpKa` (MIT), each fluorinated compound beside its
  non-fluorinated parent.
- **D3a Buchwald–Hartwig** (Ahneman et al. 2018; as redistributed in
  `SoftServeInc/yield-paper`, MIT): 3,955 reactions; the four published
  out-of-sample additive tests (sheets Test1–Test4, training rows 1–3057, 1–3055,
  1–3058 and 1–3055 respectively) and random splits.
- **D3b Suzuki–Miyaura** (Perera et al. 2018; same redistribution): 5,760 reactions,
  target `Product_Yield_PCT_Area_UV`.

## 4. Representations and base models

- Molecules: ECFP4 counts (radius 2, 2,048 bins) plus ten RDKit descriptors (MolWt,
  TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount,
  NumAromaticRings, FractionCSP3, HeavyAtomCount, NumHeteroatoms). Crippen MolLogP is
  deliberately excluded.
- Reactions: concatenated ECFP4 bit vectors (1,024 bits) of each variable component
  (BH: aryl halide, ligand, base, additive; SM: reactant 1, reactant 2, ligand,
  reagent, solvent). Component structures follow the cited redistribution.
- **M1 (primary):** random forest, 300 trees, max_features = 0.2, other settings at
  scikit-learn defaults. Spread u = standard deviation of the per-tree predictions.
- **M2 (secondary, robustness only):** ensemble of five multilayer perceptrons
  (512–128, ReLU, Adam, L2 = 1e-4, early stopping on 10% of the fit set). Spread u =
  standard deviation across members.
- Hyper-parameters are fixed here and are not tuned on any calibration or test fold.

## 5. Splits (five seeds, 0–4)

For each dataset a pool is divided by a group label: Butina clusters of ECFP4 at
Tanimoto distance 0.65 (D1), additive (D3a), ligand (D3b).

- **Held-out test H:** D1: 15% of clusters; D3a: the published TestN additives;
  D3b: two ligands per seed.
- **Held-out calibration C_h:** groups disjoint from H and from the fit set. D1: 15%
  of clusters; D3a: four additives drawn from the TestN training additives; D3b: two
  further ligands.
- **In-distribution part I** (all remaining groups) is split at random into the fit
  set F (70%), the random calibration fold C_r (15%) and the random test R (15%).
- The base model is trained on F only. Both gates (C_r-calibrated and C_h-calibrated)
  use the same model.
- **D1 external tests:** the D1 models are also scored on E1. A separate D1 model,
  trained after removing every compound with an sp³ C–F bond (SMARTS `[CX4]F`) from
  the pool, is scored on E2; fluorinated compounds and parents are reported apart.

Held-out tests are therefore: D1 cluster, D1–E1, D1–E2 (D1 family); D3a Test1–Test4
and D3b ligand (D3 family).

## 6. Signals

- **s1:** maximum Tanimoto similarity of the query fingerprint to F; **s5:** mean of
  the five largest.
- **u:** ensemble spread of the base model.
- **Conformal intervals (nominal 90%):** normalised split conformal, score
  |y − ŷ| / (u + ε), ε = 0.1 × median(u) on the calibration fold.

## 7. Gates

- Each calibration fold C is split at random into halves C_a and C_b.
- **Single-signal gates:** score s1, or −u.
- **Gate B (fused):** logistic regression of correctness on (s1, s5, log u), fitted on
  C_a. Its score is the predicted probability of a correct prediction.
- **Threshold by Learn-then-Test** on C_b, for every score: candidate thresholds are
  the score quantiles of C_b, tested in fixed sequence from the strictest that accepts
  at least 50 points of C_b, loosening one step at a time. A threshold passes when the
  binomial p-value P(Bin(n, t) ≥ k) ≤ δ, where n points are accepted and k are correct.
  The loosest threshold reached before the first failure is used. If none passes, the
  gate abstains on every query.
- **Gate A (conjunctive, no guarantee):** accept if s1 ≥ τ₁ and u ≤ τ₂; (τ₁, τ₂) from a
  20 × 20 grid of quantiles of the whole C, chosen as the largest coverage whose
  empirical precision on C is at least t.

## 8. Metrics

Per test set, model and seed: RMSE, MAE and R² ungated; error rate ungated;
risk–coverage curves of selective error rate and selective RMSE for each score;
AURC of the selective error rate (mean over all coverage levels k/n); coverage and
achieved precision of every gate at t = 0.8 and 0.9; empirical coverage of the
conformal intervals on all queries and on accepted queries. Results are averaged
over seeds; 95% intervals come from 1,000 bootstrap resamples of test queries,
pooled over seeds.

## 9. Hypotheses and pass criteria (model M1)

- **H1, the gate helps where the model is weak.** On every held-out test, the selective
  error rate of the fused score at 30% coverage is lower than the ungated error rate,
  with the 95% bootstrap interval of the difference excluding zero.
- **H2, the calibration split matters.** Over the (held-out test × target) cells,
  Gate B calibrated on C_r falls short of the target by more than 0.05 in at least
  half of the cells; calibrated on C_h, it reaches at least t − 0.05 in at least three
  quarters of the cells (seed-averaged precision).
- **H3, the combination adds something.** The AURC of Gate B's score (C_h-fitted) is at
  least 10% lower, relatively, than the better single signal on at least two held-out
  tests in each of the D1 and D3 families.
- **H4, conformal coverage under shift.** Intervals calibrated on C_r cover at least
  0.88 of R; cover less than 0.85 on at least one held-out test; and, on each held-out
  test where they fall below 0.85, cover at least 0.88 of the queries accepted by
  Gate B (C_h, t = 0.8).

Every hypothesis is reported whether or not it holds.

## 10. Exclusions

No client, proprietary or confidential data, code or results are used. Biological
context (gene expression, perturbation or cell-line shift) and generative models are
out of scope.

## Changes after this version

**1.1, 4 November 2026, before any gate or test metric was computed** (base-model
predictions were being generated; none had been scored). Three points the text above
left open are fixed as follows.
- *A gate that abstains on every query* has no defined precision. Such a seed is left
  out of the seed average. If every seed abstains, the cell is reported as
  "abstained": it counts as meeting the target in H2 (no wrong prediction is
  accepted) and is flagged in the tables.
- *H4 on a test where Gate B accepts nothing* cannot be checked; it counts as a
  failure of H4 for that test.
- *Held-out tests used for H1–H4:* D1 cluster (logp/H), D1–E1 (logp/E1) and D1–E2
  (logp_noCF/E2, all 183 compounds); D3a Test1–Test4 (bh_TestN/H) and D3b ligand
  (sm_ligand/H). The random test R of each task is the in-distribution reference.
- *Reduced training sets in D3a.* Holding out four calibration additives leaves about
  1,640 reactions in F for each TestN, roughly half of the published training rows.
  Ungated accuracy is therefore not directly comparable with published Test1–Test4
  results.

**1.2, 4 November 2026, after scoring: documentation only.** Nothing in sections 1–9
or in the pass criteria changes. Two independent reviews of the first manuscript
draft found the points below.
- *Wording versus implementation.* Section 5 says the logP held-out test and
  calibration fold receive "15% of clusters". The code (`splits.group_holdout`)
  assigns whole clusters in random order until 15% of the *compounds* is reached.
  The manuscript describes the code.
- *Suzuki–Miyaura structures.* Section 4 says the component structures "follow the
  cited redistribution". They come from a table written for this repository
  (`src/aoa/data.py`), which omits spectator fragments: the internal standard, HCl
  and water of hydration.
- *Count in note 1.1.* Note 1.1 announces three points and lists four.
- *Bootstrap correction.* The first version of `boot_h1` resampled each seed
  independently. For test sets that are identical in every seed (E1, E2, BH
  Test1–Test4), that makes the intervals too narrow. The corrected code uses the same
  resample for every seed in those sets. No H1 verdict changes; the E2 interval
  widens from [−0.064, +0.029] to [−0.098, +0.067].
- *Timing.* The whole study ran in one working session on 4 November 2026. The
  commit history shows the order: the empty repository, then protocol 1.0, then 1.1,
  then the code, then the results. Base-model predictions were generated with the
  committed code shortly before that code was committed; none were scored before
  `feadf9a`.
- *Exploratory analyses.* The following are not in the protocol and are reported as
  exploratory (`scripts/exploratory.py`):
  - per-seed target attainment;
  - pooled precision in the operating-limits cards;
  - conformal intervals calibrated on C_h;
  - the E2 signal diagnostics.


**1.3, 8 November 2026, post hoc interpretation and descriptive reporting.** The
original protocol, hypotheses, success criteria and numerical procedures above are
retained as history. This note does not retroactively prespecify a new analysis.
- The implemented binomial-test sequence is now described as an empirical rule
  inspired by Learn-then-Test. Its data-dependent candidate thresholds/order do
  not establish the previously asserted finite-sample guarantee.
- The random/held-out comparison changes scorer fitting, threshold-selection
  observations and their sizes; it compares complete pipelines, not only thresholds.
- The conformal-style rule retains the calibration-derived normalizer and original
  NumPy `higher` quantile convention. Ordinary marginal or gate-conditional coverage
  guarantees are not established for this implementation or the shifted tests.
- Pooled Wilson intervals were withdrawn because repeated query–seed observations
  are dependent. Pooled precision is descriptive. Per-seed counts, precision and
  coverage are added for both models, calibration folds and targets (SI S5–S8).
  No new population-level interval is substituted without a justified paired/group
  sampling target and statistical review. Original H1 query-bootstrap limitations
  are now explicit; its calculations and hypothesis decisions are retained.
- The 182-compound E1/E2 overlap and common BH campaign preclude interpreting the
  eight tests as eight independent replications.
- The manuscript, SI, README, figures, source package and AI-use disclosure are
  synchronized. This is an unposted working revision, not a posted version 2.
See `REVISION_NOTES.md` and `results/revision/verification.json` for verification.


**1.4, 8 November 2026, PR #1 review; no change to the experiment.** The manuscript
now distinguishes E2 seed-averaged precision (0.67) from descriptive pooled precision
(24/34, 0.71). It explicitly identifies incomplete query pairing in the original H1
bootstrap for partly overlapping seed-specific logP-H and SM-H sets. The original
H1 calculation and pass criteria are retained as history. Verification is extended
to fresh prediction replay, every report value, integer counts and complete ZIP
membership. A tie-block error in the unused `empirical_threshold` helper is fixed;
the reported LTT-labelled and conjunctive pipelines do not call that helper. No
original numerical output or hypothesis decision changes. See `PR_REVIEW.md`.
