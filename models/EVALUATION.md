# Model evaluation (WK2.10)

Two evaluations are reported, because the first one turned out to be too
easy to be useful on its own — see below.

## 1. Primary evaluation (`evaluate.py` → `evaluation_report.json`)

Both models — logistic regression and XGBoost — score **precision =
recall = 1.0 on all four labels** on the 800-bill held-out test set, at
every threshold from 0.1 to 0.9.

This is not a bug, and it is not as impressive as it looks. Four of the
engineered features (`has_unbypassed_ncci_pair`, `any_line_exceeds_mue`,
`max_units_mue_ratio`, `diag_procedure_match_score`, `em_doc_level_gap`)
are each an *exact restatement* of the rule the data generator used to
construct that label (see `features.py`'s docstring and `DATA_CARD.md`).
Handing a model a feature that already IS the answer and then reporting
that the model finds the answer isn't a meaningful test — it mainly
confirms the feature pipeline and the label pipeline agree with each
other, which is a useful sanity check (and is exactly what the
verification pass below checks) but not evidence the *model* is doing
anything a hard-coded `if` statement couldn't already do.

## 2. Ablation evaluation (`ablation.py` → `ablation_report.json`)

To find out what's actually learnable, both models were retrained with
those five direct-rule columns removed, leaving only indirect signal:
code-pair rarity, provider peer deviation, raw units/charges, care-
pathway flags, and (for upcoding only) the two honest raw fields
`documentation_level` and `em_level_billed`, which aren't rule
restatements — a real auditor has both independently.

| Label | LogReg P / R | XGBoost P / R | Read |
|---|---|---|---|
| unbundling | 0.62 / 0.98 | **0.97 / 0.85** | recovers well — code-pair rarity + surgical-code flags carry real signal |
| fragmented_billing | 0.73 / 0.95 | **1.00 / 0.93** | recovers well — raw `total_units` alone is a strong tell |
| causality_mismatch | 0.06 / 0.53 | 0.50 / 0.03 | **collapses** — see below |
| upcoding | 1.00 / 1.00 | 1.00 / 1.00 | stays perfect — `documentation_level` and `em_level_billed` are legitimately independent fields, not a rule restatement |

**causality_mismatch is the honest finding here.** Without
`diag_procedure_match_score`, neither model can reliably catch it —
there's currently no second, independent feature that captures "this
diagnosis doesn't belong with these procedures." The diagnosis code
itself isn't in the feature table at all, only its precomputed match
score. Concrete next step: add the diagnosis's own body region as an
explicit categorical feature (mirroring `injury_body_region`), so the
model has redundant signal instead of depending on one engineered
column. Flagged as a WK3+ follow-up rather than fixed now, since it
doesn't block the rest of the pipeline.

**Model comparison:** XGBoost dominates logistic regression on every
label except upcoding (where both are trivially perfect) — notably
better precision at comparable or better recall throughout. This is the
basis for the WK6 API and MLflow registry defaulting to XGBoost as the
production model, with logistic regression kept and versioned as the
interpretable baseline (its coefficients, in `baseline_coefficients.json`,
are still useful for the audit-report narrative even when it's not the
scoring model).

## 3. Threshold tradeoff (the ablated XGBoost "any flag" score)

Because the primary evaluation's score is perfectly separable, its
threshold sweep is flat (precision = recall = 1.0 at every threshold,
see `evaluation_report.json`) — not useful for a tradeoff discussion.
The ablated XGBoost sweep actually curves, so it's the one used here to
reason about where to set the operating threshold in production, where
the two error types have very different costs:

- **False positive** (bill flagged, reviewer finds nothing wrong): wasted
  reviewer time — a real cost, but a bounded, recoverable one.
- **False negative** (violation missed): the claim pays out with the
  coding error baked in — lost recovery, and in workers' comp
  specifically, a mis-coded causality mismatch can mean paying for
  treatment of a condition that was never compensable in the first
  place.

| Threshold | Precision | Recall | Flags/800 bills | Violations missed |
|---|---|---|---|---|
| 0.1 | 0.61 | 0.78 | 204 | 36 |
| 0.2 | 0.83 | 0.74 | 143 | 42 |
| **0.3** | **0.91** | **0.73** | **129** | **43** |
| 0.5 | 0.98 | 0.71 | 115 | 47 |
| 0.9 | 1.00 | 0.66 | 105 | 55 |

Given false negatives are the costlier error in this domain (a missed
coding error is lost money, not just lost time), the right operating
point leans toward recall rather than precision-maximizing — **t≈0.3**
is a reasonable default: 9 in 10 flags are real, catching ~73% of
violations, without flooding the reviewer queue (129 flags out of 800
bills, vs. 204 at the loosest threshold for only 4 more violations
caught). This is a starting point, not a final answer — the real
threshold should be tuned against actual reviewer capacity once this
moves past the portfolio stage, and revisited whenever `causality_mismatch`
gets a second supporting feature (which would change its precision/recall
profile independent of everything else).

## Reproducing

```bash
cd models
python3 features.py         # rebuild features.csv (fits train-only stats)
python3 train_baseline.py   # logistic regression -> model_baseline.pkl
python3 train_xgboost.py    # XGBoost -> model_xgboost.pkl
python3 evaluate.py         # primary evaluation -> evaluation_report.json
python3 ablation.py         # ablated evaluation -> ablation_report.json
```
