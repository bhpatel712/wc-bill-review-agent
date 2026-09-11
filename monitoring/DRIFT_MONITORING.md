# Drift monitoring & retraining trigger (WK3.2-WK3.4)

## Why this matters here specifically

Every rule table this project scores bills against is revised on a fixed
public schedule, not once and done:

- **CPT/HCPCS codes** — the AMA publishes an annual CPT code set update
  (new codes, retired codes, revised descriptions), effective January 1;
  HCPCS Level II codes are additionally updated quarterly by CMS.
- **ICD-10-CM diagnosis codes** — updated annually by CMS/NCHS, effective
  October 1.
- **NCCI PTP edit pairs** — updated quarterly by CMS.
- **MUE unit caps** — updated quarterly (some tables monthly) by CMS.

A model trained on last year's code set and billing patterns doesn't fail
loudly when the world underneath it changes — it just quietly starts
scoring bills against rules that no longer match reality: a code that was
added six months ago never appears in `min_pair_cooccurrence`'s training
counts, so the model treats it as maximally rare (suspicious) by default;
an MUE cap that was raised or lowered no longer matches what
`max_units_mue_ratio` was calibrated against. That's a genuine,
recurring reason to re-check the model on a schedule, independent of
whether anything about the *bills themselves* is unusual — this is what
WK3.1-WK3.4 build toward: a way to notice when the world has moved before
the model's silence is mistaken for confidence.

## What's built here

**WK3.1 — MLflow (`models/mlflow_utils.py`, wired into `train_baseline.py`,
`train_xgboost.py`, `evaluate.py`).** Every training run logs its
hyperparameters, the saved model file, and (via `evaluate.py`, attached to
the same run afterward) its test-set precision/recall/F1 per label. Local
SQLite-backed tracking store (`mlruns.db` at the project root) — no server
to run. Newer MLflow puts the plain-filesystem backend (`file:./mlruns`,
what most tutorials still show) into maintenance mode and refuses to write
to it; SQLite is the smallest currently-supported alternative that still
needs no server.

**WK3.2 — Drift detection (`monitoring/drift_detection.py`).** Compares a
new batch of bills against the training distribution, feature by feature,
using two standard metrics:

- **PSI (Population Stability Index)** — bins the baseline distribution
  into deciles, then checks what fraction of the new batch falls in each
  bin versus what the baseline had. `< 0.1` = no meaningful shift,
  `0.1-0.2` = moderate, `>= 0.2` = significant. Works for both numeric and
  categorical features.
- **Kolmogorov-Smirnov test** — a p-value for "these two samples came from
  the same distribution," numeric features only. Reported alongside PSI
  as a second, differently-derived opinion.

Crucially, the new batch is scored using the **same training-fit
statistics** (`compute_pair_cooccurrence`, `compute_provider_peer_stats`)
that `features.py` used to build `features.csv` — reused directly by
importing those functions, not recomputed. Refitting them fresh on the
new batch would compare the batch to a version of itself and hide real
drift; a production scoring pipeline always freezes training-time
statistics and reuses them on new data, so the demo does the same thing.

**WK3.3 — this document.**

**WK3.4 — Retraining trigger (`monitoring/retrain_trigger.py`).** Runs
drift detection, then applies a decision rule: recommend retraining only
when **at least 2 features** are independently flagged `ALERT`, not on any
single flagged feature. Logs the decision as its own MLflow run (params +
metrics + a tag listing which features triggered it), appends to a local
JSONL log, and exits with status `1` if a retrain is recommended (`0`
otherwise) — meant to be wired into a scheduled job.

## The synthetic drift scenario

To have something real to detect, `data-generation/generate_bills.py`
gained a `--dos-offset-max` flag (default 150, reproduces the original
corpus exactly — verified byte-identical before this was relied on
anywhere). Raising it shifts how long after the injury date a bill's date
of service falls, simulating a real, well-known WC pattern: **claims
staying open longer than they used to.** That single change cascades
through the generator's own logic — more bills land in the `late`
`injury_stage` bucket, which shifts `pick_pathway()`'s choices (more
physical-therapy/pain-management maintenance care, less early-stage
imaging) and therefore the downstream charge and unit distributions.

The demo batch: `--n-bills 600 --seed 99 --dos-offset-max 320
--out-subdir drift_batch` (600 bills, offset range more than doubled).

## What it actually found

Running `drift_detection.py` against that batch:

| Feature | PSI | Flag |
|---|---|---|
| days_since_injury | 0.4546 | **ALERT** |
| injury_stage | 0.2372 | **ALERT** |
| provider_specialty | 0.2107 | **ALERT** |
| max_units_mue_ratio | 0.0564 | OK |
| everything else | < 0.03 | OK |

`days_since_injury` and `injury_stage` are the real, deliberate signal —
exactly the two features `--dos-offset-max` directly and indirectly
shifts, and by a wide margin (0.45 is more than double the alert
threshold; nothing else numeric moved noticeably).

`provider_specialty` alerting is the honest complication, worth stating
plainly rather than hiding: the drift batch uses a different random seed
than the training corpus, so it draws its own fresh pool of 40 providers
— and by chance alone, a differently-seeded draw over only 5 specialty
categories at a 600-bill sample size produces some amount of proportion
drift with no causal connection to the `--dos-offset-max` scenario at
all. This is a real failure mode of single-batch drift monitoring,
especially for low-cardinality categorical features on moderate sample
sizes — which is exactly why `retrain_trigger.py`'s decision rule
requires **2 or more** independently-alerted features rather than acting
on any one. In this run it still crosses that bar (3 features alerted),
so a retrain is correctly recommended — but for the right reason
(the deliberate injury-duration shift), with the noisy feature along for
the ride rather than driving the decision by itself.

A production version of this would go further: track PSI trend across
multiple consecutive batches rather than deciding off one, and/or use a
larger current-batch sample before trusting a categorical PSI. Flagged as
a real next step, not fixed here, since it doesn't block demonstrating
the mechanism end to end.

## Reproducing

```bash
# from data-generation/
python3 generate_bills.py --n-bills 600 --seed 99 --dos-offset-max 320 --out-subdir drift_batch

# from models/ -- now also logs params/metrics/artifacts to MLflow
python3 train_baseline.py
python3 train_xgboost.py
python3 evaluate.py

# from monitoring/
python3 drift_detection.py
python3 retrain_trigger.py   # exit code 1 == retrain recommended

# optional: browse the MLflow UI (run from the project root)
mlflow ui --backend-store-uri sqlite:///mlruns.db
# then open http://127.0.0.1:5000
```
