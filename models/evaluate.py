"""
evaluate.py  

Evaluates both models on the held-out test split: per-label precision /
recall / F1, plus a derived "any flag" view, plus a threshold sweep on
that derived score to make the false-positive/false-negative tradeoff
concrete (see EVALUATION.md for the write-up).
"""

import json
import os
import pickle

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

from dataset import load_encoded

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(HERE, "evaluation_report.json")


def any_flag_proba(proba_per_label):
    """proba_per_label: list of arrays (n_samples,) of P(label=1), one per label.
    Returns P(at least one label fires), assuming approximate independence."""
    p_none = np.ones_like(proba_per_label[0])
    for p in proba_per_label:
        p_none *= (1 - p)
    return 1 - p_none


def get_probas(clf, X):
    # MultiOutputClassifier.predict_proba returns a list (one per label) of
    # (n_samples, 2) arrays; take the P(class=1) column from each.
    raw = clf.predict_proba(X)
    return [p[:, 1] for p in raw]


def evaluate_model(name, model_path, X_test, y_test, label_names, needs_scaling=False, scaler=None):
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    clf = saved["model"]

    X_eval = scaler.transform(X_test) if needs_scaling else X_test
    probas = get_probas(clf, X_eval)  # list of per-label P(1)
    preds = [(p >= 0.5).astype(int) for p in probas]

    per_label = {}
    for i, label in enumerate(label_names):
        per_label[label] = {
            "precision": round(float(precision_score(y_test[:, i], preds[i], zero_division=0)), 4),
            "recall": round(float(recall_score(y_test[:, i], preds[i], zero_division=0)), 4),
            "f1": round(float(f1_score(y_test[:, i], preds[i], zero_division=0)), 4),
            "positives_in_test": int(y_test[:, i].sum()),
            "positives_predicted": int(sum(preds[i])),
        }

    y_any = (y_test.sum(axis=1) > 0).astype(int)
    p_any = any_flag_proba(probas)
    pred_any = (p_any >= 0.5).astype(int)
    any_flag = {
        "precision": round(float(precision_score(y_any, pred_any, zero_division=0)), 4),
        "recall": round(float(recall_score(y_any, pred_any, zero_division=0)), 4),
        "f1": round(float(f1_score(y_any, pred_any, zero_division=0)), 4),
    }

    # threshold sweep on the derived "any flag" score, for the tradeoff write-up
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep = []
    for t in thresholds:
        pred_t = (p_any >= t).astype(int)
        sweep.append({
            "threshold": t,
            "precision": round(float(precision_score(y_any, pred_t, zero_division=0)), 4),
            "recall": round(float(recall_score(y_any, pred_t, zero_division=0)), 4),
            "flags_raised": int(pred_t.sum()),
            "missed_violations": int(((pred_t == 0) & (y_any == 1)).sum()),
        })

    return {"per_label": per_label, "any_flag": any_flag, "threshold_sweep": sweep}


def main():
    d = load_encoded()
    X_test, y_test, label_names = d["X_test"], d["y_test"], d["label_names"]

    with open(os.path.join(HERE, "model_baseline.pkl"), "rb") as f:
        baseline_saved = pickle.load(f)
    scaler = baseline_saved["scaler"]

    results = {
        "test_set_size": len(X_test),
        "test_set_violation_rate": round(float((y_test.sum(axis=1) > 0).mean()), 4),
        "logistic_regression": evaluate_model(
            "logistic_regression", os.path.join(HERE, "model_baseline.pkl"),
            X_test, y_test, label_names, needs_scaling=True, scaler=scaler,
        ),
        "xgboost": evaluate_model(
            "xgboost", os.path.join(HERE, "model_xgboost.pkl"),
            X_test, y_test, label_names, needs_scaling=False,
        ),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nsaved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
