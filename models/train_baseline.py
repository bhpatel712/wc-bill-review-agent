"""
train_baseline.py  (WK2.7)

Baseline model: logistic regression, one per violation type (via
MultiOutputClassifier — see MODEL_FRAMING.md for why multi-label).
Chosen as the baseline specifically because its coefficients are directly
interpretable: for a compliance/audit use case, being able to say "this
feature pushed the score up by X" matters as much as raw accuracy.
"""

import json
import os
import pickle

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

from dataset import load_encoded
from mlflow_utils import init_experiment, record_run_id

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model_baseline.pkl")
COEF_PATH = os.path.join(HERE, "baseline_coefficients.json")


def main():
    d = load_encoded()
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]

    init_experiment()
    with mlflow.start_run(run_name="baseline_logistic_regression") as run:
        params = {
            "model_type": "logistic_regression",
            "max_iter": 2000,
            "class_weight": "balanced",
            "n_features": len(d["feature_names"]),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        mlflow.log_params(params)

        # logistic regression benefits from scaled inputs; tree models (WK2.8) don't need this
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        base = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf = MultiOutputClassifier(base)
        clf.fit(X_train_scaled, y_train)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": clf, "scaler": scaler, "feature_names": d["feature_names"], "label_names": d["label_names"]}, f)

        # dump coefficients per label for interpretability
        coef_report = {}
        for label_name, estimator in zip(d["label_names"], clf.estimators_):
            coefs = sorted(
                zip(d["feature_names"], estimator.coef_[0]),
                key=lambda kv: -abs(kv[1]),
            )
            coef_report[label_name] = [{"feature": f, "coef": round(float(c), 4)} for f, c in coefs[:8]]

        with open(COEF_PATH, "w") as f:
            json.dump(coef_report, f, indent=2)

        # artifacts: the pickled model + the interpretability report, attached
        # to this run so a later "which run produced model_baseline.pkl?"
        # question always has an answer
        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(COEF_PATH)
        record_run_id("logistic_regression", run.info.run_id)

        print(f"saved model -> {MODEL_PATH}")
        print(f"saved top coefficients -> {COEF_PATH}")
        print(json.dumps(coef_report, indent=2))
        print(f"\nMLflow run: {run.info.run_id}  (experiment: wc-bill-review-agent)")


if __name__ == "__main__":
    main()
