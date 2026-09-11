"""
train_xgboost.py  (WK2.8)

Production model: gradient-boosted trees (XGBoost), one per violation
type via MultiOutputClassifier. Trees handle the non-linear / threshold
style rules in this problem (e.g. "units > MUE cap", "gap > 1 level")
more naturally than a linear model, and don't require feature scaling.
"""

import json
import os
import pickle

import mlflow
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from dataset import load_encoded
from mlflow_utils import init_experiment, record_run_id

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model_xgboost.pkl")
IMPORTANCE_PATH = os.path.join(HERE, "xgboost_importances.json")


def main():
    d = load_encoded()
    X_train, y_train = d["X_train"], d["y_train"]

    init_experiment()
    with mlflow.start_run(run_name="xgboost") as run:
        hyperparams = dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
        mlflow.log_params({
            "model_type": "xgboost",
            **hyperparams,
            "n_features": len(d["feature_names"]),
            "n_train": len(X_train),
        })

        base = XGBClassifier(**hyperparams)
        clf = MultiOutputClassifier(base)
        clf.fit(X_train, y_train)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": clf, "feature_names": d["feature_names"], "label_names": d["label_names"]}, f)

        importance_report = {}
        for label_name, estimator in zip(d["label_names"], clf.estimators_):
            importances = sorted(
                zip(d["feature_names"], estimator.feature_importances_),
                key=lambda kv: -kv[1],
            )
            importance_report[label_name] = [{"feature": f, "importance": round(float(v), 4)} for f, v in importances[:8]]

        with open(IMPORTANCE_PATH, "w") as f:
            json.dump(importance_report, f, indent=2)

        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(IMPORTANCE_PATH)
        record_run_id("xgboost", run.info.run_id)

        print(f"saved model -> {MODEL_PATH}")
        print(f"saved feature importances -> {IMPORTANCE_PATH}")
        print(json.dumps(importance_report, indent=2))
        print(f"\nMLflow run: {run.info.run_id}  (experiment: wc-bill-review-agent)")


if __name__ == "__main__":
    main()
