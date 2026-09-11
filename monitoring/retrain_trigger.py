"""
retrain_trigger.py  (WK3.4)

Turns drift_detection.py's per-feature PSI/KS report into a go/no-go
retraining decision, and records that decision to MLflow as its own
lightweight run -- so "did we check for drift, and what did we decide"
gets the same audit trail a training run gets.

Decision rule (deliberately conservative -- see DRIFT_MONITORING.md):
recommend retraining only when at least MIN_ALERT_FEATURES features are
independently in ALERT (PSI >= 0.2), not on a single alerted feature.
Running drift_detection.py against the demo drift batch shows why this
matters: injury_stage and days_since_injury alert for a real, deliberate
reason (the batch simulates claims staying open longer), but
provider_specialty also alerts, purely from sampling noise in a
moderate-sized, differently-seeded batch drawing from only 5 categories.
Triggering a retrain off any single alerted feature would mean retraining
on noise as often as on real drift; requiring more than one independent
feature to corroborate is a cheap way to cut that down.

Run:
    python3 retrain_trigger.py
    python3 retrain_trigger.py --batch ../data-generation/output/drift_batch/synthetic_bills.jsonl

Exit code is 0 when no retrain is recommended, 1 when it is -- so this can
be wired into a scheduled job (cron, GitHub Actions, etc.) that pages a
human or kicks off retraining when it fails.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import mlflow

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))

from mlflow_utils import init_experiment  # noqa: E402
from drift_detection import run as run_drift_detection, DEFAULT_BATCH_PATH  # noqa: E402

MIN_ALERT_FEATURES = 2
LOG_PATH = os.path.join(HERE, "retrain_trigger_log.jsonl")


def decide(drift_report):
    alerted = [f for f, v in drift_report["features"].items() if v["flag"] == "ALERT"]
    recommend = len(alerted) >= MIN_ALERT_FEATURES
    return recommend, alerted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=DEFAULT_BATCH_PATH)
    args = ap.parse_args()

    drift_report = run_drift_detection(args.batch)
    recommend, alerted = decide(drift_report)

    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch": os.path.relpath(args.batch, HERE),
        "n_alert_features": len(alerted),
        "alerted_features": alerted,
        "min_alert_features_required": MIN_ALERT_FEATURES,
        "retrain_recommended": recommend,
    }

    # append-only log: every check this trigger has ever made, kept even
    # across runs, so a history of drift checks accumulates over time
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(decision) + "\n")

    init_experiment()
    run_name = f"drift_check_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("batch", decision["batch"])
        mlflow.log_param("min_alert_features_required", MIN_ALERT_FEATURES)
        mlflow.log_metric("n_alert_features", decision["n_alert_features"])
        mlflow.log_metric("retrain_recommended", int(recommend))
        mlflow.set_tag("alerted_features", ",".join(alerted) or "none")

    print(json.dumps(decision, indent=2))
    if recommend:
        print(f"\nRETRAIN RECOMMENDED -- {len(alerted)} feature(s) in ALERT: {', '.join(alerted)}")
        sys.exit(1)
    else:
        print(f"\nno retrain recommended ({decision['n_alert_features']} feature(s) alerted, "
              f"below the {MIN_ALERT_FEATURES}-feature corroboration threshold)")
        sys.exit(0)


if __name__ == "__main__":
    main()
