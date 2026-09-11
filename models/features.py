"""
features.py 

Builds the engineered feature table from the synthetic bill corpus. Two
passes: first over the whole corpus to compute corpus-level statistics
(code-pair co-occurrence frequency, per-provider peer norms), then per
bill to assemble the feature row. Output is consumed by train_baseline.py
and train_xgboost.py.

Run:
    python3 features.py
"""

import csv
import itertools
import json
import os
import random
import statistics
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BILLS_PATH = os.path.join(HERE, "..", "data-generation", "output", "synthetic_bills.jsonl")
REF_DIR = os.path.join(HERE, "..", "data-generation", "reference")
OUT_PATH = os.path.join(HERE, "features.csv")

VIOLATION_TYPES = ["unbundling", "fragmented_billing", "causality_mismatch", "upcoding"]

EM_CODES = {"99202": 1, "99203": 2, "99204": 3, "99205": 4,
            "99212": 1, "99213": 2, "99214": 3, "99215": 4}

# same region pools used by the generator, duplicated here deliberately:
# feature engineering should treat this as domain knowledge available at
# scoring time, not something derived from the corpus itself.
REGION_CPT_POOL = {
    "spine-lumbar": {"97110", "97112", "97140", "97530", "97535", "72110", "72148"},
    "spine-cervical": {"97110", "97112", "97140", "97530", "97535"},
    "shoulder": {"97110", "97112", "97140", "29826", "29827", "23472", "23600", "20610", "73030"},
    "knee": {"97110", "97112", "97140", "97530", "27447", "29877", "29880", "29881", "20610", "73721"},
    "wrist-hand": {"97110", "97112", "97140", "97535", "25607", "26720", "20605", "29125", "20670", "20680"},
    "hip": {"97110", "97112", "97530", "27130", "29862", "20610", "73721"},
    "ankle-foot": {"97110", "97112", "97116", "97140", "27758", "20605", "29405"},
    "head": set(),
    "elbow": {"97110", "97112", "97140", "20605", "29075"},
}

SURGICAL_CODES = {"23472", "27130", "27447", "29826", "29827", "29877", "29880",
                   "29881", "29862", "25607", "26720", "27758", "23600", "24500", "24505"}
PT_CODES = {"97110", "97112", "97116", "97140", "97530", "97535", "97124"}
IMAGING_CODES = {"72110", "73030", "73721", "72148"}
INJECTION_CODES = {"20550", "20605", "20610"}


def load_bills():
    with open(BILLS_PATH) as f:
        return [json.loads(l) for l in f]


def load_mue():
    mue = {}
    with open(os.path.join(REF_DIR, "mue_table.csv")) as f:
        for row in csv.DictReader(f):
            mue[row["code"]] = int(row["mue_units"])
    return mue


def load_icd_region_lookup():
    lookup = {}
    with open(os.path.join(REF_DIR, "icd10_codes.csv")) as f:
        for row in csv.DictReader(f):
            lookup[row["code"]] = row["body_region"]
    return lookup


def load_ncci_pairs():
    pairs = []
    with open(os.path.join(REF_DIR, "ncci_ptp_edits.csv")) as f:
        for row in csv.DictReader(f):
            pairs.append((row["column1_code"], row["column2_code"], row["modifier_indicator"]))
    return pairs


# ---------------------------------------------------------------------------
# Train/test split, decided BEFORE any corpus-level statistic is computed.
# ---------------------------------------------------------------------------

def stratified_split(bills, test_frac=0.2, seed=13):
    """Split by bill_id, stratified on the label (clean vs. each violation
    type) so every label is represented at its true ratio in both splits.
    Returns a set of bill_ids assigned to train."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for b in bills:
        key = b["anomaly_types"][0] if b["anomaly_types"] else "clean"
        strata[key].append(b["bill_id"])

    train_ids = set()
    for key, ids in strata.items():
        ids = ids[:]
        rng.shuffle(ids)
        n_test = round(len(ids) * test_frac)
        train_ids.update(ids[n_test:])
    return train_ids


# ---------------------------------------------------------------------------
# Pass 1: corpus-level statistics — fit on TRAIN bills only, so nothing
# about the test split leaks into a feature value (verification).
# ---------------------------------------------------------------------------

def compute_pair_cooccurrence(bills):
    counts = Counter()
    for b in bills:
        codes = sorted({l["cpt_code"] for l in b["lines"]})
        for a, c in itertools.combinations(codes, 2):
            counts[(a, c)] += 1
    return counts


def compute_provider_peer_stats(bills):
    """Per-provider average total_billed and lines/bill, z-scored against
    the provider's own specialty peer group (WK2.3)."""
    by_provider = defaultdict(list)
    specialty_of = {}
    for b in bills:
        by_provider[b["provider_id"]].append(b)
        specialty_of[b["provider_id"]] = b["provider_specialty"]

    provider_avg_charge = {}
    provider_avg_lines = {}
    for pid, pbills in by_provider.items():
        provider_avg_charge[pid] = statistics.mean(b["total_billed"] for b in pbills)
        provider_avg_lines[pid] = statistics.mean(len(b["lines"]) for b in pbills)

    by_specialty = defaultdict(list)
    for pid, specialty in specialty_of.items():
        by_specialty[specialty].append(pid)

    charge_z, lines_z = {}, {}
    for specialty, pids in by_specialty.items():
        charges = [provider_avg_charge[p] for p in pids]
        lines = [provider_avg_lines[p] for p in pids]
        c_mean, c_std = statistics.mean(charges), (statistics.pstdev(charges) or 1.0)
        l_mean, l_std = statistics.mean(lines), (statistics.pstdev(lines) or 1.0)
        for p in pids:
            charge_z[p] = (provider_avg_charge[p] - c_mean) / c_std
            lines_z[p] = (provider_avg_lines[p] - l_mean) / l_std

    return charge_z, lines_z


# ---------------------------------------------------------------------------
# Pass 2: per-bill feature row
# ---------------------------------------------------------------------------

def build_feature_row(bill, pair_counts, charge_z, lines_z, mue, icd_region, ncci_pairs):
    lines = bill["lines"]
    codes = [l["cpt_code"] for l in lines]
    distinct_codes = sorted(set(codes))

    # code-pair co-occurrence: rarest pair present on this bill
    if len(distinct_codes) >= 2:
        freqs = [pair_counts.get((a, c), 0) for a, c in itertools.combinations(distinct_codes, 2)]
        min_pair_cooccurrence = min(freqs)
    else:
        min_pair_cooccurrence = -1  # sentinel: no pair exists on this bill

    # modifier usage
    num_modifier59_lines = sum(1 for l in lines if l.get("modifier") == "59")
    # does the bill contain both codes of a *known* NCCI pair with no bypass modifier?
    has_unbypassed_ncci_pair = 0
    for col1, col2, indicator in ncci_pairs:
        if col1 in codes and col2 in codes:
            col2_lines = [l for l in lines if l["cpt_code"] == col2]
            bypassed = indicator == "1" and any(l.get("modifier") == "59" for l in col2_lines)
            if not bypassed:
                has_unbypassed_ncci_pair = 1
                break

    # provider peer deviation
    provider_charge_z = charge_z.get(bill["provider_id"], 0.0)
    provider_lines_z = lines_z.get(bill["provider_id"], 0.0)

    # units vs MUE ratio
    ratios = [l["units"] / mue[l["cpt_code"]] for l in lines if l["cpt_code"] in mue]
    max_units_mue_ratio = max(ratios) if ratios else 0.0
    any_line_exceeds_mue = 1 if any(r > 1.0 for r in ratios) else 0

    # diagnosis-procedure category match score
    diag_regions = {icd_region.get(d) for d in bill["diagnosis_codes"]}
    proc_regions = set()
    for c in distinct_codes:
        for region, pool in REGION_CPT_POOL.items():
            if c in pool:
                proc_regions.add(region)
    if not proc_regions:
        diag_procedure_match_score = 1.0  # no region-specific procedure billed (e.g. E/M only) -> nothing to mismatch
    elif diag_regions & proc_regions:
        diag_procedure_match_score = 1.0
    else:
        diag_procedure_match_score = 0.0

    # WK2.6 time-since-injury vs procedure type
    has_surgical = 1 if any(c in SURGICAL_CODES for c in distinct_codes) else 0
    has_pt = 1 if any(c in PT_CODES for c in distinct_codes) else 0
    has_imaging = 1 if any(c in IMAGING_CODES for c in distinct_codes) else 0
    has_injection = 1 if any(c in INJECTION_CODES for c in distinct_codes) else 0
    early_stage_surgery_flag = 1 if (has_surgical and bill["injury_stage"] == "early") else 0

    # E/M / documentation-level features
    em_levels_billed = [EM_CODES[c] for c in distinct_codes if c in EM_CODES]
    em_level_billed = max(em_levels_billed) if em_levels_billed else 0
    em_doc_level_gap = em_level_billed - bill["documentation_level"] if em_level_billed else 0

    row = {
        "bill_id": bill["bill_id"],
        "split": bill["_split"],
        "total_billed": bill["total_billed"],
        "num_lines": len(lines),
        "total_units": sum(l["units"] for l in lines),
        "documentation_level": bill["documentation_level"],
        "is_new_patient": int(bill["is_new_patient"]),
        "days_since_injury": bill["days_since_injury"],
        "injury_stage": bill["injury_stage"],
        "provider_specialty": bill["provider_specialty"],
        "injury_body_region": bill["injury_body_region"],

        "min_pair_cooccurrence": min_pair_cooccurrence,
        "num_modifier59_lines": num_modifier59_lines,
        "has_unbypassed_ncci_pair": has_unbypassed_ncci_pair,

        "provider_charge_zscore": round(provider_charge_z, 4),
        "provider_lines_zscore": round(provider_lines_z, 4),

        "max_units_mue_ratio": round(max_units_mue_ratio, 4),
        "any_line_exceeds_mue": any_line_exceeds_mue,

        "diag_procedure_match_score": diag_procedure_match_score,

        "has_surgical_code": has_surgical,
        "has_pt_code": has_pt,
        "has_imaging_code": has_imaging,
        "has_injection_code": has_injection,
        "early_stage_surgery_flag": early_stage_surgery_flag,

        "em_level_billed": em_level_billed,
        "em_doc_level_gap": em_doc_level_gap,

        # labels (multi-label + binary-any)
        "label_unbundling": int("unbundling" in bill["anomaly_types"]),
        "label_fragmented_billing": int("fragmented_billing" in bill["anomaly_types"]),
        "label_causality_mismatch": int("causality_mismatch" in bill["anomaly_types"]),
        "label_upcoding": int("upcoding" in bill["anomaly_types"]),
        "label_any": int(bill["is_anomalous"]),
    }
    return row


def main():
    bills = load_bills()
    mue = load_mue()
    icd_region = load_icd_region_lookup()
    ncci_pairs = load_ncci_pairs()

    train_ids = stratified_split(bills, test_frac=0.2, seed=13)
    for b in bills:
        b["_split"] = "train" if b["bill_id"] in train_ids else "test"
    train_bills = [b for b in bills if b["_split"] == "train"]

    # fit corpus-level statistics on the training split only
    pair_counts = compute_pair_cooccurrence(train_bills)
    charge_z, lines_z = compute_provider_peer_stats(train_bills)

    rows = [build_feature_row(b, pair_counts, charge_z, lines_z, mue, icd_region, ncci_pairs) for b in bills]

    fieldnames = list(rows[0].keys())
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_test = len(rows) - n_train
    print(f"wrote {len(rows)} feature rows -> {OUT_PATH}  (train={n_train}, test={n_test})")
    print(f"columns: {fieldnames}")
    print("NOTE: code-pair co-occurrence and provider peer z-scores are fit on the "
          "train split only, then applied to score both splits — this avoids leaking "
          "test-set statistics into the features.")


if __name__ == "__main__":
    main()
