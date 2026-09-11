"""
inference.py  (WK6 support)

Wraps the trained Week 2 XGBoost model for serving. Loads once at process
startup (ModelService.__init__) and reuses models/features.py's exact
feature-engineering functions -- not a reimplementation of them -- so a
bill scored through this API gets identically-computed features to one
scored during training/evaluation.

One real wrinkle this has to handle: two of features.py's corpus-level
statistics (code-pair co-occurrence counts, provider peer z-scores) are
*fit on the training split*, not looked up per-bill. To score a brand new
bill the same way evaluate.py scored the held-out test split, this
refits those same statistics from the same training bills (same seed,
same stratified_split as features.py/dataset.py use) at startup, then
reuses them for every request -- refitting per-request would be wasteful
and, worse, would let a single incoming bill's own stats leak into its
own score.
"""
import os
import sys
import pickle

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "..", "models")
sys.path.insert(0, MODELS_DIR)

import features as features_mod  # noqa: E402  (models/features.py)
from dataset import CATEGORICAL_COLS  # noqa: E402  (models/dataset.py)

MODEL_PATH = os.path.join(MODELS_DIR, "model_xgboost.pkl")

# violation_type (what the rest of this project calls it) -> label column
# name (what features.py/dataset.py call it). Kept in one place so the API
# layer speaks "violation_type" everywhere, matching agent/retrieve.py.
LABEL_COL_FOR = {
    "unbundling": "label_unbundling",
    "fragmented_billing": "label_fragmented_billing",
    "causality_mismatch": "label_causality_mismatch",
    "upcoding": "label_upcoding",
}


class ModelNotReadyError(RuntimeError):
    """Raised when the trained model or its supporting reference data
    isn't available -- e.g. models/model_xgboost.pkl hasn't been trained
    yet (see README.md step 3), or the Week 1 reference tables/corpus
    haven't been built (step 1-2)."""


class ModelService:
    def __init__(self):
        try:
            bills = features_mod.load_bills()
            mue = features_mod.load_mue()
            icd_region = features_mod.load_icd_region_lookup()
            ncci_pairs = features_mod.load_ncci_pairs()

            train_ids = features_mod.stratified_split(bills, test_frac=0.2, seed=13)
            train_bills = [b for b in bills if b["bill_id"] in train_ids]

            self.pair_counts = features_mod.compute_pair_cooccurrence(train_bills)
            self.charge_z, self.lines_z = features_mod.compute_provider_peer_stats(train_bills)
            self.mue = mue
            self.icd_region = icd_region
            self.ncci_pairs = ncci_pairs

            with open(MODEL_PATH, "rb") as f:
                saved = pickle.load(f)
            self.clf = saved["model"]
            self.feature_names = saved["feature_names"]
            self.label_names = saved["label_names"]

            # Keyed lookup over the full synthetic corpus (ground-truth
            # fields included) -- backs the UI's "look up a bill by ID"
            # flow (GET /bills/{bill_id} in main.py) so a demo doesn't
            # require the user to hand-craft bill JSON. This is corpus
            # data already loaded above for feature-fitting, just also
            # kept indexed by id; it costs one dict, not a second load.
            self.bills_by_id = {b["bill_id"]: b for b in bills}
        except FileNotFoundError as e:
            raise ModelNotReadyError(
                f"Missing a required file ({e.filename}). Run the Week 1-2 pipeline "
                "first (see README.md's \"Running it end to end\", steps 1-3) before "
                "starting this API."
            ) from e

    def predict(self, bill: dict) -> dict:
        """bill: a plain dict matching BillIn's fields (schemas.py). Returns
        {violation_type: {"probability": float, "flagged": bool}, ...}."""
        # build_feature_row() (features.py) expects a full corpus-style bill
        # record, including a few fields that only exist for training
        # (_split, anomaly_types, is_anomalous). Those all end up in columns
        # dataset.py drops before scoring (DROP_COLS), so their VALUES don't
        # matter here -- these are placeholders purely to satisfy
        # build_feature_row's dict access, not real inputs to the model.
        bill_for_features = dict(bill)
        bill_for_features["_split"] = "serve"
        bill_for_features.setdefault("anomaly_types", [])
        bill_for_features.setdefault("is_anomalous", False)

        row = features_mod.build_feature_row(
            bill_for_features, self.pair_counts, self.charge_z, self.lines_z,
            self.mue, self.icd_region, self.ncci_pairs,
        )

        df = pd.DataFrame([row])
        encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS)
        # reindex to the exact column set/order the model was trained on --
        # a category value never seen in training (e.g. an unfamiliar
        # provider_specialty) just produces all-zero dummy columns for that
        # field, the same fallback pandas' own get_dummies gives an unseen
        # category at serving time. fill_value=0 is what makes that safe.
        encoded = encoded.reindex(columns=self.feature_names, fill_value=0)
        X = encoded.astype(float).values

        raw = self.clf.predict_proba(X)  # list of (1, 2) arrays, one per label, same order as self.label_names
        probs = {label: float(p[0, 1]) for label, p in zip(self.label_names, raw)}

        predictions = {}
        for violation_type, label_col in LABEL_COL_FOR.items():
            p = round(probs[label_col], 4)
            predictions[violation_type] = {"probability": p, "flagged": p >= 0.5}
        return predictions
