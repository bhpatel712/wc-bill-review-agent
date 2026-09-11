"""
dataset.py

Shared train/test loading + encoding for train_baseline.py, train_xgboost.py,
and evaluate.py. Reads features.csv (already split into train/test by
features.py) and returns encoded matrices.
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(HERE, "features.csv")

LABEL_COLS = ["label_unbundling", "label_fragmented_billing", "label_causality_mismatch", "label_upcoding"]
CATEGORICAL_COLS = ["injury_stage", "provider_specialty", "injury_body_region"]
DROP_COLS = ["bill_id", "split", "label_any"] + LABEL_COLS


# The four features that are each an exact restatement of the rule used
# to construct one label (see features.py). Included by default because
# the spec calls for them and a real system would have them, but they
# make the primary evaluation trivially perfect (see EVALUATION.md) —
# ablation.py drops these to see what's learnable from indirect signal alone.
DIRECT_RULE_COLS = ["has_unbypassed_ncci_pair", "any_line_exceeds_mue",
                     "max_units_mue_ratio", "diag_procedure_match_score", "em_doc_level_gap"]


def load_encoded(drop_direct_rule_features=False):
    df = pd.read_csv(FEATURES_PATH)
    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS)
    exclude = set(DROP_COLS) | (set(DIRECT_RULE_COLS) if drop_direct_rule_features else set())
    feature_cols = [c for c in encoded.columns if c not in exclude]

    train = encoded[df["split"] == "train"]
    test = encoded[df["split"] == "test"]

    X_train = train[feature_cols].astype(float).values
    X_test = test[feature_cols].astype(float).values
    y_train = train[LABEL_COLS].astype(int).values
    y_test = test[LABEL_COLS].astype(int).values

    bill_ids_test = df[df["split"] == "test"]["bill_id"].values

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_cols,
        "label_names": LABEL_COLS,
        "bill_ids_test": bill_ids_test,
    }


if __name__ == "__main__":
    d = load_encoded()
    print("X_train", d["X_train"].shape, "y_train", d["y_train"].shape)
    print("X_test ", d["X_test"].shape, "y_test ", d["y_test"].shape)
    print("features:", d["feature_names"])
