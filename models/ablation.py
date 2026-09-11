"""
ablation.py  (supplementary)

The primary evaluation (evaluate.py) scores ~1.0 precision/recall on
every label. That's not a bug — four of the engineered features
(has_unbypassed_ncci_pair, any_line_exceeds_mue, max_units_mue_ratio,
diag_procedure_match_score, em_doc_level_gap) are each an exact
restatement of the rule used to construct that label, so the model has
essentially been handed the answer. That's a legitimate thing to ship
(a real system has these fields too, per the spec's own feature list),
but it means the primary evaluation doesn't tell us whether the model
adds anything over just hard-coding the rule checks directly.

This script retrains both model types with those five columns removed,
to see what's recoverable from indirect / statistical signal alone
(code-pair rarity, provider peer deviation, raw units/level magnitudes,
care-pathway flags). Results feed the write-up in EVALUATION.md.
"""

import json
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from dataset import load_encoded

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(HERE, "ablation_report.json")


def run(model_name):
    d = load_encoded(drop_direct_rule_features=True)
    X_train, y_train = d["X_train"], d["y_train"]
    X_test, y_test = d["X_test"], d["y_test"]

    if model_name == "logistic_regression":
        scaler = StandardScaler()
        X_train_use = scaler.fit_transform(X_train)
        X_test_use = scaler.transform(X_test)
        clf = MultiOutputClassifier(LogisticRegression(max_iter=2000, class_weight="balanced"))
    else:
        X_train_use, X_test_use = X_train, X_test
        clf = MultiOutputClassifier(XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42,
        ))

    clf.fit(X_train_use, y_train)
    raw = clf.predict_proba(X_test_use)
    probas = [p[:, 1] for p in raw]
    preds = [(p >= 0.5).astype(int) for p in probas]

    per_label = {}
    for i, label in enumerate(d["label_names"]):
        per_label[label] = {
            "precision": round(float(precision_score(y_test[:, i], preds[i], zero_division=0)), 4),
            "recall": round(float(recall_score(y_test[:, i], preds[i], zero_division=0)), 4),
            "f1": round(float(f1_score(y_test[:, i], preds[i], zero_division=0)), 4),
        }

    # "any flag" derived score + threshold sweep -- with direct-rule features
    # removed, this boundary is no longer trivially separable, so the sweep
    # actually shows a precision/recall tradeoff (unlike evaluate.py's).
    import numpy as np
    p_none = np.ones_like(probas[0])
    for p in probas:
        p_none *= (1 - p)
    p_any = 1 - p_none
    y_any = (y_test.sum(axis=1) > 0).astype(int)
    sweep = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred_t = (p_any >= t).astype(int)
        sweep.append({
            "threshold": t,
            "precision": round(float(precision_score(y_any, pred_t, zero_division=0)), 4),
            "recall": round(float(recall_score(y_any, pred_t, zero_division=0)), 4),
            "flags_raised": int(pred_t.sum()),
            "missed_violations": int(((pred_t == 0) & (y_any == 1)).sum()),
        })

    return per_label, d["feature_names"], sweep


def main():
    results = {}
    for model_name in ["logistic_regression", "xgboost"]:
        per_label, feature_names, sweep = run(model_name)
        results[model_name] = {"per_label": per_label, "any_flag_threshold_sweep": sweep}
    results["_features_used"] = feature_names
    results["_features_dropped"] = ["has_unbypassed_ncci_pair", "any_line_exceeds_mue",
                                     "max_units_mue_ratio", "diag_procedure_match_score", "em_doc_level_gap"]

    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nsaved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
