"""
mlflow_utils.py  (WK3.1)

Shared MLflow setup, used by every script in this directory so they all
log to the same local experiment store on disk (../mlruns/), regardless
of which directory the script happens to be run from.

This project uses MLflow's local file-based tracking store -- no server
to stand up, appropriate for a solo/portfolio project. A team would set
MLFLOW_TRACKING_URI to point at a shared server instead; nothing else
here would need to change.

Design note: train_baseline.py and train_xgboost.py each start one MLflow
run per training call and log params + the saved model artifact. They do
NOT log test-set metrics themselves -- that would mean duplicating the
precision/recall/F1 logic that already lives in evaluate.py, and the two
copies would eventually drift out of sync. Instead each training script
records its run_id here (record_run_id), and evaluate.py looks it up
(get_run_id) and logs its metrics into that SAME run after the fact. One
run per trained model, metrics attached once, one source of truth for how
those metrics are computed.

Tracking store note: newer MLflow (3.x) puts its plain-filesystem tracking
backend ("file:./mlruns") into maintenance mode and refuses to write to it
unless you explicitly opt back in -- it wants a database backend instead.
This uses a local SQLite file (mlruns.db) for that reason: still nothing
to install or run as a server, but on the currently-supported path rather
than the deprecated one.
"""
import json
import os

import mlflow

HERE = os.path.dirname(os.path.abspath(__file__))
TRACKING_DB_PATH = os.path.abspath(os.path.join(HERE, "..", "mlruns.db"))
RUN_REGISTRY_PATH = os.path.join(HERE, "last_runs.json")


def init_experiment(name="wc-bill-review-agent"):
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB_PATH}")
    mlflow.set_experiment(name)


def record_run_id(model_key, run_id):
    registry = {}
    if os.path.exists(RUN_REGISTRY_PATH):
        with open(RUN_REGISTRY_PATH) as f:
            registry = json.load(f)
    registry[model_key] = run_id
    with open(RUN_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def get_run_id(model_key):
    if not os.path.exists(RUN_REGISTRY_PATH):
        return None
    with open(RUN_REGISTRY_PATH) as f:
        return json.load(f).get(model_key)
