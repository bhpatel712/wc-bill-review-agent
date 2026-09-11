"""
drift_detection.py  (WK3.2)

Compares the feature distribution of a NEW ("current") batch of bills
against the distribution the models were trained on, using two standard
drift metrics:

  - PSI (Population Stability Index): a single number per feature summarizing
    how much a distribution has shifted, using industry-standard cutoffs
    (< 0.1 = no meaningful shift, 0.1-0.2 = moderate, > 0.2 = significant).
  - The two-sample Kolmogorov-Smirnov test: a p-value for "these two samples
    were drawn from the same distribution," for numeric features only.

Both are reported per feature so nothing is hidden behind one aggregate
score -- a real reviewer needs to know WHICH feature moved, not just that
"something" did.

Important: this reuses features.py's own feature-building code, and feeds
it the SAME training-fit statistics (code-pair co-occurrence counts,
provider peer z-score norms) that went into features.csv. Recomputing
those fresh on the new batch would defeat the entire point -- you'd be
comparing the new batch to a version of itself, not to what the model was
actually trained on. This mirrors a real production rule: whatever
statistics you fit at training time get frozen and reused at scoring
time; you never refit them on the data you're trying to score.

Run:
    python3 drift_detection.py
    python3 drift_detection.py --batch ../data-generation/output/drift_batch/synthetic_bills.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats as scipy_stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))

from features import (  # noqa: E402
    load_bills, load_mue, load_icd_region_lookup, load_ncci_pairs,
    stratified_split, compute_pair_cooccurrence, compute_provider_peer_stats,
    build_feature_row,
)

DEFAULT_BATCH_PATH = os.path.join(HERE, "..", "data-generation", "output", "drift_batch", "synthetic_bills.jsonl")
REPORT_PATH = os.path.join(HERE, "drift_report.json")

NUMERIC_FEATURES = [
    "total_billed", "num_lines", "total_units", "documentation_level",
    "days_since_injury", "min_pair_cooccurrence", "num_modifier59_lines",
    "provider_charge_zscore", "provider_lines_zscore",
    "max_units_mue_ratio", "em_level_billed",
]
CATEGORICAL_FEATURES = ["injury_stage", "provider_specialty", "injury_body_region"]

PSI_BINS = 10
PSI_WARN, PSI_ALERT = 0.1, 0.2


def psi(expected, actual, bins=PSI_BINS):
    """Population Stability Index between two 1-D numeric arrays. Bin edges
    come from the EXPECTED (baseline/training) distribution's own deciles --
    both samples are then binned against those same fixed edges. That's the
    standard PSI recipe, and it's why direction matters: psi(a, b) != psi(b, a)
    in general, because the bin edges themselves are asymmetric."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0  # near-constant feature in the baseline -- nothing meaningful to bin
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = np.clip(e_counts / max(len(expected), 1), 1e-4, None)
    a_pct = np.clip(a_counts / max(len(actual), 1), 1e-4, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_categorical(expected, actual):
    """Same PSI formula, bucketed by category value instead of quantile bins."""
    cats = sorted(set(expected) | set(actual))
    e_pct = np.clip(np.array([expected.count(c) / max(len(expected), 1) for c in cats]), 1e-4, None)
    a_pct = np.clip(np.array([actual.count(c) / max(len(actual), 1) for c in cats]), 1e-4, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def flag_for(psi_score):
    if psi_score >= PSI_ALERT:
        return "ALERT"
    if psi_score >= PSI_WARN:
        return "WARN"
    return "OK"


def build_baseline_and_current(batch_path):
    bills = load_bills()
    mue = load_mue()
    icd_region = load_icd_region_lookup()
    ncci_pairs = load_ncci_pairs()

    # the exact same split, and the exact same corpus-level stats, that
    # features.py fit when it built features.csv -- seed=13 makes this
    # reproducible rather than something we'd have to re-derive by hand.
    train_ids = stratified_split(bills, test_frac=0.2, seed=13)
    train_bills = [b for b in bills if b["bill_id"] in train_ids]
    pair_counts = compute_pair_cooccurrence(train_bills)
    charge_z, lines_z = compute_provider_peer_stats(train_bills)

    baseline_rows = [
        build_feature_row({**b, "_split": "train"}, pair_counts, charge_z, lines_z, mue, icd_region, ncci_pairs)
        for b in train_bills
    ]

    with open(batch_path) as f:
        current_bills = [json.loads(line) for line in f]

    # score the NEW batch with the frozen training-fit stats -- never refit on it
    current_rows = [
        build_feature_row({**b, "_split": "current"}, pair_counts, charge_z, lines_z, mue, icd_region, ncci_pairs)
        for b in current_bills
    ]
    return baseline_rows, current_rows


def run(batch_path):
    baseline_rows, current_rows = build_baseline_and_current(batch_path)

    report = {
        "baseline": "training split (features.py, seed=13)",
        "current_batch": os.path.relpath(batch_path, HERE),
        "n_baseline": len(baseline_rows),
        "n_current": len(current_rows),
        "features": {},
    }

    for feat in NUMERIC_FEATURES:
        expected = np.array([r[feat] for r in baseline_rows], dtype=float)
        actual = np.array([r[feat] for r in current_rows], dtype=float)
        psi_score = psi(expected, actual)
        ks_stat, ks_p = scipy_stats.ks_2samp(expected, actual)
        report["features"][feat] = {
            "type": "numeric",
            "psi": round(psi_score, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 6),
            "baseline_mean": round(float(expected.mean()), 3),
            "current_mean": round(float(actual.mean()), 3),
            "flag": flag_for(psi_score),
        }

    for feat in CATEGORICAL_FEATURES:
        expected = [r[feat] for r in baseline_rows]
        actual = [r[feat] for r in current_rows]
        psi_score = psi_categorical(expected, actual)
        report["features"][feat] = {
            "type": "categorical",
            "psi": round(psi_score, 4),
            "flag": flag_for(psi_score),
        }

    n_alert = sum(1 for f in report["features"].values() if f["flag"] == "ALERT")
    n_warn = sum(1 for f in report["features"].values() if f["flag"] == "WARN")
    report["summary"] = {
        "n_features_checked": len(report["features"]),
        "n_alert": n_alert,
        "n_warn": n_warn,
        "overall": "ALERT" if n_alert else ("WARN" if n_warn else "OK"),
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=DEFAULT_BATCH_PATH)
    ap.add_argument("--out", default=REPORT_PATH)
    args = ap.parse_args()

    report = run(args.batch)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
