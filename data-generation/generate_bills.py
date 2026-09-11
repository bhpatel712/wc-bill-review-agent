"""
generate_bills.py

Synthetic workers' comp bill generator (WK1.6-1.8 of the build plan).

Builds a labeled corpus of fully synthetic WC bills: synthetic patient,
synthetic provider, injury type, date of service, CPT/ICD codes billed,
and a synthetic billed amount. ~80% of bills are constructed "clean"
(pass every rule in the reference DB); ~20% are constructed by taking a
clean bill and deliberately injecting exactly one of four violation types,
so every anomalous bill carries a ground-truth label of *which* rule it
breaks and why. Nothing here is drawn from, or resembles, any real
patient, provider, or claim.

Violation types injected (see NOTES_AND_GUARDRAILS.md / DATA_CARD.md):
  - unbundling            : bills both codes of a known NCCI PTP edit pair
                             on the same date of service, no modifier.
  - fragmented_billing    : bills units for one code above its MUE threshold.
  - causality_mismatch    : diagnosis body region doesn't match the billed
                             procedures' body region (procedure isn't
                             plausibly related to the injury).
  - upcoding              : bills an E/M code whose complexity level exceeds
                             what the synthetic documentation level supports.

Usage:
    python3 generate_bills.py --n-bills 4000 --seed 42
"""

import argparse
import csv
import json
import os
import random
import sqlite3
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "wc_bill_review.db")
OUT_DIR = os.path.join(HERE, "output")

VIOLATION_TYPES = ["unbundling", "fragmented_billing", "causality_mismatch", "upcoding"]

BODY_REGIONS = [
    "spine-lumbar", "spine-cervical", "shoulder", "knee",
    "wrist-hand", "hip", "ankle-foot", "head", "elbow",
]

# region -> list of CPT codes plausibly billed for that region, beyond E/M.
# (kept disjoint from the "other region" pool on purpose, so a
# causality-mismatch mutation can cleanly swap in a code from elsewhere.)
REGION_CPT_POOL = {
    "spine-lumbar": ["97110", "97112", "97140", "97530", "97535", "72110", "72148"],
    "spine-cervical": ["97110", "97112", "97140", "97530", "97535"],
    "shoulder": ["97110", "97112", "97140", "29826", "29827", "23472", "23600", "20610", "73030"],
    "knee": ["97110", "97112", "97140", "97530", "27447", "29877", "29880", "29881", "20610", "73721"],
    "wrist-hand": ["97110", "97112", "97140", "97535", "25607", "26720", "20605", "29125", "20670", "20680"],
    "hip": ["97110", "97112", "97530", "27130", "29862", "20610", "73721"],
    "ankle-foot": ["97110", "97112", "97116", "97140", "27758", "20605", "29405"],
    "head": [],
    "elbow": ["97110", "97112", "97140", "20605", "29075"],
}

# E/M complexity level (1-5) -> (new-patient code, established-patient code)
EM_LEVELS = {
    1: ("99202", "99212"),
    2: ("99203", "99213"),
    3: ("99204", "99214"),
    4: ("99205", "99215"),
    5: ("99205", "99215"),  # level 5 reuses the top code; documentation is what differs
}

PT_TIMED_CODES = {"97110", "97112", "97116", "97140", "97530", "97535", "97124"}

PROVIDER_SPECIALTIES = [
    "Occupational Medicine", "Orthopedic Surgery", "Physical Therapy",
    "Primary Care", "Pain Management",
]

# Synthetic base fee per code (dollars); actual line charge jitters +/-15%.
BASE_FEE = {
    "99202": 140, "99203": 190, "99204": 260, "99205": 340,
    "99212": 90, "99213": 130, "99214": 190, "99215": 260,
    "99221": 210, "99231": 110,
    "97110": 55, "97112": 60, "97116": 58, "97140": 62, "97530": 58,
    "97535": 50, "97124": 45,
    "23472": 3200, "27130": 3600, "27447": 3800,
    "29826": 1450, "29827": 2100, "29877": 900, "29880": 1500, "29881": 1300,
    "29862": 2400, "25607": 1900, "26720": 700, "27758": 2600,
    "23600": 320, "24500": 340, "24505": 420,
    "20550": 95, "20605": 110, "20610": 130,
    "29075": 180, "29125": 120, "29405": 190,
    "20670": 260, "20680": 480,
    "72110": 95, "73030": 85, "73721": 620, "72148": 640,
}


def load_reference(conn):
    cur = conn.cursor()
    cur.execute("SELECT code, description, category, unit_basis FROM cpt_hcpcs_codes")
    cpt = {r[0]: {"description": r[1], "category": r[2], "unit_basis": r[3]} for r in cur.fetchall()}

    cur.execute("SELECT code, description, body_region FROM icd10_codes")
    icd_rows = cur.fetchall()
    icd_by_region = {}
    for code, desc, region in icd_rows:
        icd_by_region.setdefault(region, []).append({"code": code, "description": desc})

    cur.execute("SELECT column1_code, column2_code, modifier_indicator, rationale, source FROM ncci_ptp_edits")
    ncci_edits = [
        {"col1": r[0], "col2": r[1], "modifier_indicator": r[2], "rationale": r[3], "source": r[4]}
        for r in cur.fetchall()
    ]

    cur.execute("SELECT code, mue_units, rationale FROM mue_table")
    mue = {r[0]: {"units": r[1], "rationale": r[2]} for r in cur.fetchall()}

    return cpt, icd_by_region, ncci_edits, mue


def make_patient_pool(n, rng):
    return [f"WC-PT-{i:05d}" for i in range(1, n + 1)]


def make_provider_pool(n, rng):
    providers = []
    for i in range(1, n + 1):
        specialty = rng.choice(PROVIDER_SPECIALTIES)
        providers.append({
            "provider_id": f"WC-PR-{i:04d}",
            "specialty": specialty,
            # a per-provider billing "aggressiveness" multiplier used later
            # to give the peer-deviation feature (WK2.3) something real to find
            "billing_intensity": round(rng.gauss(1.0, 0.12), 3),
        })
    return providers


def em_code_for_level(level, is_new_patient):
    new_code, est_code = EM_LEVELS[level]
    return new_code if is_new_patient else est_code


def days_since_injury_bucket(days):
    if days <= 14:
        return "early"
    if days <= 60:
        return "mid"
    return "late"


def pick_pathway(rng, region, specialty, stage):
    """Choose a care pathway consistent with provider specialty and injury stage."""
    pool = REGION_CPT_POOL.get(region, [])
    surgical_codes = [c for c in pool if BASE_FEE.get(c, 0) >= 700 and c not in PT_TIMED_CODES]
    pt_codes = [c for c in pool if c in PT_TIMED_CODES]
    other_codes = [c for c in pool if c not in PT_TIMED_CODES and c not in surgical_codes]

    if specialty == "Physical Therapy" and pt_codes:
        return "pt_only", pt_codes
    if specialty == "Orthopedic Surgery" and surgical_codes and stage != "early":
        return "surgery_episode", surgical_codes
    if specialty == "Pain Management" and other_codes:
        injections = [c for c in other_codes if c.startswith("20")]
        if injections:
            return "injection", injections
    if stage == "early" and other_codes:
        imaging = [c for c in other_codes if c.startswith("7")]
        if imaging:
            return "imaging", imaging
    if pt_codes:
        return "em_plus_pt", pt_codes
    return "em_only", []


def build_clean_bill(bill_id, rng, patients, providers, cpt, icd_by_region, mue, ncci_edits, start_date, span_days, dos_offset_max=150):
    region = rng.choice(BODY_REGIONS)
    while not icd_by_region.get(region):
        region = rng.choice(BODY_REGIONS)

    patient_id = rng.choice(patients)
    provider = rng.choice(providers)
    diag = rng.choice(icd_by_region[region])

    date_of_injury = start_date + timedelta(days=rng.randrange(span_days))
    dos_offset = rng.randrange(0, dos_offset_max)
    date_of_service = date_of_injury + timedelta(days=dos_offset)
    stage = days_since_injury_bucket(dos_offset)

    pathway, code_pool = pick_pathway(rng, region, provider["specialty"], stage)

    # documentation level: how much clinical complexity the note actually
    # supports (1 = brief, 5 = extensive). Clean bills always bill an E/M
    # code that matches this level.
    doc_level = rng.choices([1, 2, 3, 4, 5], weights=[15, 30, 30, 18, 7])[0]
    is_new_patient = dos_offset < 10 and rng.random() < 0.6

    lines = []

    include_em = pathway in ("em_only", "em_plus_pt", "injection", "imaging") or rng.random() < 0.5
    if include_em:
        em_code = em_code_for_level(doc_level, is_new_patient)
        lines.append({"cpt_code": em_code, "units": 1, "modifier": ""})

    if pathway == "pt_only":
        n_codes = rng.choice([1, 2, 2, 3])
        chosen = rng.sample(code_pool, min(n_codes, len(code_pool)))
        for c in chosen:
            cap = mue.get(c, {}).get("units", 4)
            lines.append({"cpt_code": c, "units": rng.randint(1, max(1, cap - 1)), "modifier": ""})
    elif pathway == "em_plus_pt":
        n_codes = rng.choice([1, 2])
        chosen = rng.sample(code_pool, min(n_codes, len(code_pool)))
        for c in chosen:
            cap = mue.get(c, {}).get("units", 4)
            lines.append({"cpt_code": c, "units": rng.randint(1, max(1, cap - 1)), "modifier": ""})
    elif pathway == "surgery_episode":
        # one primary procedure code; a second, NCCI-paired code is added
        # below (with or without a bypass modifier) only in the deliberate
        # bypass-pattern branch, never accidentally here.
        c = rng.choice(code_pool)
        lines.append({"cpt_code": c, "units": 1, "modifier": ""})
    elif pathway in ("injection", "imaging"):
        c = rng.choice(code_pool)
        cap = mue.get(c, {}).get("units", 1)
        lines.append({"cpt_code": c, "units": rng.randint(1, max(1, cap)), "modifier": ""})
    # em_only: nothing further to add

    if not lines:
        # head-region / no pool fallback: always at least an E/M line
        em_code = em_code_for_level(doc_level, is_new_patient)
        lines.append({"cpt_code": em_code, "units": 1, "modifier": ""})

    # A PT/OT pathway can legitimately sample both codes of a known NCCI
    # pair (e.g. 97110 + 97140). Real billing has two valid outcomes here:
    # document the two as distinct procedures and append modifier 59 to
    # the bundled code, or just don't bill both. Model both, weighted
    # toward the simpler "don't bill both" case, so the modifier field
    # carries real signal instead of always being empty.
    codes_on_bill = {l["cpt_code"] for l in lines}
    for edit in ncci_edits:
        if edit["col1"] in codes_on_bill and edit["col2"] in codes_on_bill:
            if edit["modifier_indicator"] == "1" and rng.random() < 0.35:
                for l in lines:
                    if l["cpt_code"] == edit["col2"]:
                        l["modifier"] = "59"
            else:
                lines = [l for l in lines if l["cpt_code"] != edit["col2"]]
            break  # one pair handled is enough for a single bill

    for line in lines:
        base = BASE_FEE.get(line["cpt_code"], 100)
        jitter = rng.uniform(0.85, 1.15) * provider["billing_intensity"]
        line["charge"] = round(base * jitter * line["units"], 2)

    total_billed = round(sum(l["charge"] for l in lines), 2)

    bill = {
        "bill_id": bill_id,
        "patient_id": patient_id,
        "provider_id": provider["provider_id"],
        "provider_specialty": provider["specialty"],
        "injury_body_region": region,
        "date_of_injury": date_of_injury.isoformat(),
        "date_of_service": date_of_service.isoformat(),
        "days_since_injury": dos_offset,
        "injury_stage": stage,
        "is_new_patient": is_new_patient,
        "documentation_level": doc_level,
        "diagnosis_codes": [diag["code"]],
        "lines": lines,
        "total_billed": total_billed,
        "is_anomalous": False,
        "anomaly_types": [],
        "anomaly_details": [],
    }
    return bill


def violates_any_rule(bill, ncci_edits, mue, icd_by_region_lookup):
    """Ground-truth checker used to (a) keep 'clean' bills genuinely clean,
    and (b) confirm a mutated bill trips exactly the rule it was meant to."""
    codes_on_bill = {l["cpt_code"] for l in bill["lines"]}
    hits = []

    lines_by_code = {l["cpt_code"]: l for l in bill["lines"]}
    for edit in ncci_edits:
        if edit["col1"] in codes_on_bill and edit["col2"] in codes_on_bill:
            col2_line = lines_by_code.get(edit["col2"], {})
            has_bypass_modifier = col2_line.get("modifier") == "59"
            # indicator "0" pairs are never separately payable, modifier or not
            if edit["modifier_indicator"] == "0" or not has_bypass_modifier:
                hits.append(("unbundling", edit))

    for line in bill["lines"]:
        cap = mue.get(line["cpt_code"], {}).get("units")
        if cap is not None and line["units"] > cap:
            hits.append(("fragmented_billing", line))

    diag_regions = {icd_by_region_lookup.get(d) for d in bill["diagnosis_codes"]}
    proc_regions = set()
    for l in bill["lines"]:
        for region, pool in REGION_CPT_POOL.items():
            if l["cpt_code"] in pool:
                proc_regions.add(region)
    if proc_regions and diag_regions and not (diag_regions & proc_regions):
        hits.append(("causality_mismatch", {"diag_regions": diag_regions, "proc_regions": proc_regions}))

    for line in bill["lines"]:
        if line["cpt_code"] in {"99202", "99203", "99204", "99205", "99212", "99213", "99214", "99215"}:
            level = next((lvl for lvl, (nc, ec) in EM_LEVELS.items() if line["cpt_code"] in (nc, ec)), None)
            if level is not None and level > bill["documentation_level"] + 1:
                hits.append(("upcoding", line))

    return hits


def inject_violation(bill, violation_type, rng, cpt, icd_by_region, ncci_edits, mue, icd_region_lookup):
    bill = json.loads(json.dumps(bill))  # deep copy
    detail = None

    if violation_type == "unbundling":
        candidates = [e for e in ncci_edits if e["col1"] in cpt and e["col2"] in cpt]
        region_pool = REGION_CPT_POOL.get(bill["injury_body_region"], [])
        same_region = [e for e in candidates if e["col1"] in region_pool and e["col2"] in region_pool]
        edit = rng.choice(same_region or candidates)
        bill["lines"] = [l for l in bill["lines"] if l["cpt_code"] not in (edit["col1"], edit["col2"])]
        for code in (edit["col1"], edit["col2"]):
            base = BASE_FEE.get(code, 400)
            bill["lines"].append({"cpt_code": code, "units": 1, "modifier": "", "charge": round(base * rng.uniform(0.9, 1.1), 2)})
        detail = f"Billed {edit['col1']} with {edit['col2']} on the same date of service without a bypass modifier. {edit['rationale']} (NCCI PTP, source={edit['source']})"

    elif violation_type == "fragmented_billing":
        eligible = [l for l in bill["lines"] if l["cpt_code"] in mue]
        if not eligible:
            code = rng.choice(list(mue.keys()))
            base = BASE_FEE.get(code, 100)
            line = {"cpt_code": code, "units": 1, "modifier": "", "charge": round(base * rng.uniform(0.9, 1.1), 2)}
            bill["lines"].append(line)
        else:
            line = rng.choice(eligible)
        cap = mue[line["cpt_code"]]["units"]
        excess = rng.randint(2, 6)
        line["units"] = cap + excess
        line["charge"] = round(BASE_FEE.get(line["cpt_code"], 100) * line["units"] * rng.uniform(0.95, 1.05), 2)
        detail = f"Billed {line['units']} units of {line['cpt_code']} in one day; MUE threshold is {cap}. {mue[line['cpt_code']]['rationale']}"

    elif violation_type == "causality_mismatch":
        proc_regions = set()
        for l in bill["lines"]:
            for region, pool in REGION_CPT_POOL.items():
                if l["cpt_code"] in pool:
                    proc_regions.add(region)
        other_regions = [r for r in BODY_REGIONS if r not in proc_regions and icd_by_region.get(r)]
        if not other_regions:
            other_regions = [r for r in BODY_REGIONS if icd_by_region.get(r)]
        new_region = rng.choice(other_regions)
        new_diag = rng.choice(icd_by_region[new_region])
        old_diag = bill["diagnosis_codes"][0]
        bill["diagnosis_codes"] = [new_diag["code"]]
        detail = f"Procedures billed are consistent with the '{'/'.join(sorted(proc_regions)) or bill['injury_body_region']}' region, but the diagnosis was changed to {new_diag['code']} ({new_diag['description']}), region '{new_region}' — not causally related to the billed treatment."
        bill["original_diagnosis_code"] = old_diag

    elif violation_type == "upcoding":
        em_lines = [l for l in bill["lines"] if l["cpt_code"] in {v for pair in EM_LEVELS.values() for v in pair}]
        bill["documentation_level"] = rng.choice([1, 2])
        top_code = em_code_for_level(5, bill["is_new_patient"])
        if em_lines:
            old_code = em_lines[0]["cpt_code"]
            em_lines[0]["cpt_code"] = top_code
            em_lines[0]["charge"] = round(BASE_FEE.get(top_code, 300) * rng.uniform(0.95, 1.05), 2)
        else:
            old_code = None
            bill["lines"].append({"cpt_code": top_code, "units": 1, "modifier": "", "charge": round(BASE_FEE.get(top_code, 300) * rng.uniform(0.95, 1.05), 2)})
        detail = f"Billed {top_code} (highest E/M complexity level) but synthetic documentation level is {bill['documentation_level']}/5 — supports at most a level {bill['documentation_level'] + 1} visit."

    bill["total_billed"] = round(sum(l["charge"] for l in bill["lines"]), 2)
    bill["is_anomalous"] = True
    bill["anomaly_types"] = [violation_type]
    bill["anomaly_details"] = [detail]
    return bill


def generate(n_bills, violation_rate, seed, start_year=2025, dos_offset_max=150):
    rng = random.Random(seed)
    conn = sqlite3.connect(DB_PATH)
    cpt, icd_by_region, ncci_edits, mue = load_reference(conn)
    conn.close()

    icd_region_lookup = {}
    for region, rows in icd_by_region.items():
        for r in rows:
            icd_region_lookup[r["code"]] = region

    patients = make_patient_pool(600, rng)
    providers = make_provider_pool(40, rng)
    start_date = date(start_year, 1, 1)

    n_violation = round(n_bills * violation_rate)
    n_clean = n_bills - n_violation
    per_type = n_violation // len(VIOLATION_TYPES)
    violation_plan = []
    for vt in VIOLATION_TYPES:
        violation_plan += [vt] * per_type
    while len(violation_plan) < n_violation:
        violation_plan.append(rng.choice(VIOLATION_TYPES))
    rng.shuffle(violation_plan)

    bills = []
    bill_id = 1

    # clean bills
    made_clean = 0
    attempts = 0
    while made_clean < n_clean and attempts < n_clean * 20:
        attempts += 1
        bill = build_clean_bill(f"WC-BILL-{bill_id:06d}", rng, patients, providers, cpt, icd_by_region, mue, ncci_edits, start_date, 300, dos_offset_max=dos_offset_max)
        if violates_any_rule(bill, ncci_edits, mue, icd_region_lookup):
            continue  # regenerate; don't burn an id on a rejected draft
        bills.append(bill)
        bill_id += 1
        made_clean += 1

    # violation bills: start from a fresh clean base, then mutate
    made_violation = 0
    attempts = 0
    while made_violation < n_violation and attempts < n_violation * 20:
        attempts += 1
        vt = violation_plan[made_violation]
        base = build_clean_bill(f"WC-BILL-{bill_id:06d}", rng, patients, providers, cpt, icd_by_region, mue, ncci_edits, start_date, 300, dos_offset_max=dos_offset_max)
        if violates_any_rule(base, ncci_edits, mue, icd_region_lookup):
            continue
        mutated = inject_violation(base, vt, rng, cpt, icd_by_region, ncci_edits, mue, icd_region_lookup)
        hits = violates_any_rule(mutated, ncci_edits, mue, icd_region_lookup)
        hit_types = {h[0] for h in hits}
        if hit_types != {vt}:
            continue  # mutation accidentally tripped 0 or >1 rules; discard and retry
        bills.append(mutated)
        bill_id += 1
        made_violation += 1

    rng.shuffle(bills)
    return bills, {
        "requested_n_bills": n_bills,
        "actual_n_bills": len(bills),
        "requested_violation_rate": violation_rate,
        "actual_n_violation": sum(1 for b in bills if b["is_anomalous"]),
        "actual_n_clean": sum(1 for b in bills if not b["is_anomalous"]),
        "violation_type_counts": {
            vt: sum(1 for b in bills if vt in b["anomaly_types"]) for vt in VIOLATION_TYPES
        },
        "seed": seed,
        "dos_offset_max": dos_offset_max,
    }


def write_outputs(bills, stats, out_dir=None):
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "synthetic_bills.jsonl")
    with open(jsonl_path, "w") as f:
        for b in bills:
            f.write(json.dumps(b) + "\n")

    csv_path = os.path.join(out_dir, "synthetic_bills_flat.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bill_id", "patient_id", "provider_id", "provider_specialty",
            "injury_body_region", "date_of_injury", "date_of_service",
            "days_since_injury", "injury_stage", "documentation_level",
            "diagnosis_codes", "cpt_lines", "total_billed",
            "is_anomalous", "anomaly_types", "anomaly_details",
        ])
        for b in bills:
            cpt_lines = "; ".join(
                f"{l['cpt_code']}{'-' + l['modifier'] if l.get('modifier') else ''}x{l['units']}(${l['charge']})"
                for l in b["lines"]
            )
            writer.writerow([
                b["bill_id"], b["patient_id"], b["provider_id"], b["provider_specialty"],
                b["injury_body_region"], b["date_of_injury"], b["date_of_service"],
                b["days_since_injury"], b["injury_stage"], b["documentation_level"],
                "|".join(b["diagnosis_codes"]), cpt_lines, b["total_billed"],
                b["is_anomalous"], "|".join(b["anomaly_types"]), " ~~ ".join(b["anomaly_details"]),
            ])

    stats_path = os.path.join(out_dir, "generation_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"wrote {len(bills)} bills -> {jsonl_path}")
    print(f"wrote flat view          -> {csv_path}")
    print(f"wrote generation stats   -> {stats_path}")
    print(json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bills", type=int, default=4000)
    ap.add_argument("--violation-rate", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    # WK3.2: --dos-offset-max shifts how long after injury a bill's date of
    # service falls. Default 150 reproduces the original corpus exactly.
    # A larger value simulates claims staying open longer -- more bills
    # land in the "late" injury_stage bucket, which cascades into a
    # different care-pathway mix (more PT/pain-management, less early
    # imaging) and different charge/unit distributions. Used to build a
    # synthetic "current batch" for drift_detection.py.
    ap.add_argument("--dos-offset-max", type=int, default=150)
    # writes into output/<out-subdir>/ instead of output/ directly, so a
    # drift-scenario batch never overwrites the main training corpus.
    ap.add_argument("--out-subdir", type=str, default="")
    args = ap.parse_args()

    bills, stats = generate(args.n_bills, args.violation_rate, args.seed, dos_offset_max=args.dos_offset_max)
    out_dir = os.path.join(OUT_DIR, args.out_subdir) if args.out_subdir else OUT_DIR
    write_outputs(bills, stats, out_dir=out_dir)


if __name__ == "__main__":
    main()
