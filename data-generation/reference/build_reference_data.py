"""
build_reference_data.py

Builds the ground-truth rule engine reference tables for the WC Bill Review
Agent: CPT/HCPCS codes, ICD-10-CM codes, NCCI Procedure-to-Procedure (PTP)
edit pairs, and Medically Unlikely Edit (MUE) unit thresholds.

SCOPE & SOURCING NOTE (read this before treating any of this as authoritative):

CMS publishes the full NCCI PTP edit files and MUE tables quarterly as
downloadable spreadsheets covering the entire CPT/HCPCS code set (hundreds
of thousands of edit pairs, tens of thousands of MUE rows). This build
environment has no network access to cms.gov, so this script does NOT parse
a live CMS file. Instead it hand-curates a workers'-comp-relevant SUBSET:

  - CPT/HCPCS code numbers and category are real, public codes commonly
    billed in WC claims (E/M, physical/occupational therapy, orthopedic
    surgery, injections, casting, radiology). Descriptions here are original
    short paraphrases written for this project, NOT copied from the
    licensed AMA CPT manual.
  - ICD-10-CM codes are real, fully public, CMS-published codes for common
    workplace injuries.
  - NCCI_PTP_EDITS rows marked source="cms_documented" reflect long-standing,
    widely-documented bundling relationships (e.g. subacromial decompression
    into rotator cuff repair, chondroplasty into meniscectomy, manual therapy
    vs. therapeutic exercise). Rows marked source="illustrative_pattern" are
    additional pairs constructed to follow the SAME real edit patterns
    (same-joint/same-session procedure bundling) so the synthetic generator
    has enough variety; they are plausible but not verified against a
    specific current-quarter CMS file.
  - MUE_TABLE unit thresholds follow long-documented CMS conventions for
    these code families (most discrete surgical procedures = 1/day, most
    15-minute timed therapy codes ~= 4/day, E/M = 1/day). Exact current-
    quarter values can shift by a unit or two.

Before using this for anything beyond a portfolio demo, replace this file's
output with a parse of the actual current CMS NCCI PTP / MUE quarterly
release (same column shapes are used here on purpose, to make that swap
mechanical).
"""

import csv
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# CPT / HCPCS reference codes
# ---------------------------------------------------------------------------
# code, description (original short paraphrase), category, typical_units_note

CPT_HCPCS_CODES = [
    # Evaluation & Management
    ("99202", "New patient office visit, straightforward decision making", "E/M", "per encounter"),
    ("99203", "New patient office visit, low complexity decision making", "E/M", "per encounter"),
    ("99204", "New patient office visit, moderate complexity decision making", "E/M", "per encounter"),
    ("99205", "New patient office visit, high complexity decision making", "E/M", "per encounter"),
    ("99212", "Established patient office visit, straightforward decision making", "E/M", "per encounter"),
    ("99213", "Established patient office visit, low complexity decision making", "E/M", "per encounter"),
    ("99214", "Established patient office visit, moderate complexity decision making", "E/M", "per encounter"),
    ("99215", "Established patient office visit, high complexity decision making", "E/M", "per encounter"),
    ("99221", "Initial hospital care, low complexity", "E/M", "per encounter"),
    ("99231", "Subsequent hospital care, low complexity", "E/M", "per encounter"),

    # Physical / Occupational Therapy (timed, 15-minute units)
    ("97110", "Therapeutic exercise for strength, endurance, or range of motion", "PT/OT", "15-min unit"),
    ("97112", "Neuromuscular reeducation of movement, balance, and coordination", "PT/OT", "15-min unit"),
    ("97116", "Gait training therapy", "PT/OT", "15-min unit"),
    ("97140", "Manual therapy techniques, one or more regions", "PT/OT", "15-min unit"),
    ("97530", "Therapeutic activities, direct patient contact", "PT/OT", "15-min unit"),
    ("97535", "Self-care / home management training", "PT/OT", "15-min unit"),
    ("97124", "Massage therapy, including effleurage and petrissage", "PT/OT", "15-min unit"),

    # Orthopedic Surgery
    ("23472", "Total shoulder arthroplasty", "Surgery-Ortho", "per procedure"),
    ("27130", "Total hip arthroplasty", "Surgery-Ortho", "per procedure"),
    ("27447", "Total knee arthroplasty", "Surgery-Ortho", "per procedure"),
    ("29826", "Arthroscopic subacromial decompression, shoulder", "Surgery-Ortho", "per procedure"),
    ("29827", "Arthroscopic rotator cuff repair, shoulder", "Surgery-Ortho", "per procedure"),
    ("29877", "Arthroscopic chondroplasty, knee", "Surgery-Ortho", "per procedure"),
    ("29880", "Arthroscopic meniscectomy, medial and lateral, knee", "Surgery-Ortho", "per procedure"),
    ("29881", "Arthroscopic meniscectomy, single compartment, knee", "Surgery-Ortho", "per procedure"),
    ("29862", "Hip arthroscopy with labral repair", "Surgery-Ortho", "per procedure"),
    ("25607", "Open reduction internal fixation, distal radius fracture", "Surgery-Ortho", "per procedure"),
    ("26720", "Closed treatment of phalangeal shaft fracture with percutaneous fixation", "Surgery-Ortho", "per digit"),
    ("27758", "Open reduction internal fixation, tibial shaft fracture", "Surgery-Ortho", "per procedure"),

    # Fracture care (closed treatment)
    ("23600", "Closed treatment of shoulder fracture, without manipulation", "Fracture-Care", "per procedure"),
    ("24500", "Closed treatment of humeral shaft fracture, without manipulation", "Fracture-Care", "per procedure"),
    ("24505", "Closed treatment of humeral shaft fracture, with manipulation", "Fracture-Care", "per procedure"),

    # Injections / aspirations
    ("20550", "Injection of tendon sheath or ligament", "Injection", "per site"),
    ("20605", "Arthrocentesis, aspiration/injection, intermediate joint", "Injection", "per joint"),
    ("20610", "Arthrocentesis, aspiration/injection, major joint", "Injection", "per joint"),

    # Casting / splinting
    ("29075", "Application of short arm cast", "Cast-Splint", "per limb"),
    ("29125", "Application of short arm splint, static", "Cast-Splint", "per limb"),
    ("29405", "Application of short leg cast", "Cast-Splint", "per limb"),

    # Implant removal
    ("20670", "Removal of superficial implant (e.g. pin, wire)", "Surgery-Ortho", "per procedure"),
    ("20680", "Removal of deep implant (e.g. buried plate, screw, rod)", "Surgery-Ortho", "per procedure"),

    # Radiology
    ("72110", "X-ray, lumbar spine, 4 or more views", "Radiology", "per study"),
    ("73030", "X-ray, shoulder, 2 or more views", "Radiology", "per study"),
    ("73721", "MRI, lower extremity joint, without contrast", "Radiology", "per study"),
    ("72148", "MRI, lumbar spine, without contrast", "Radiology", "per study"),
]

CPT_HCPCS_HEADER = ["code", "description", "category", "unit_basis"]


# ---------------------------------------------------------------------------
# ICD-10-CM reference codes (workplace-injury relevant)
# ---------------------------------------------------------------------------
# code, description, body_region  (body_region drives the causality-match feature)

ICD10_CODES = [
    ("S39.012A", "Strain of muscle, fascia, and tendon of lower back, initial encounter", "spine-lumbar"),
    ("M54.50", "Low back pain, unspecified", "spine-lumbar"),
    ("S33.5XXA", "Sprain of ligaments of lumbar spine, initial encounter", "spine-lumbar"),
    ("S13.4XXA", "Sprain of ligaments of cervical spine, initial encounter", "spine-cervical"),

    ("S43.401A", "Sprain of unspecified rotator cuff capsule, initial encounter", "shoulder"),
    ("M75.100", "Unspecified rotator cuff tear or rupture, not specified as traumatic", "shoulder"),
    ("S42.001A", "Displaced fracture of unspecified part of clavicle, initial encounter", "shoulder"),

    ("S83.511A", "Sprain of anterior cruciate ligament of right knee, initial encounter", "knee"),
    ("S83.281A", "Other tear of medial meniscus, current injury, right knee, initial encounter", "knee"),
    ("M25.561", "Pain in right knee", "knee"),

    ("S62.001A", "Unspecified displaced fracture of navicular bone of right wrist, initial encounter", "wrist-hand"),
    ("S61.409A", "Unspecified open wound of right hand, initial encounter", "wrist-hand"),
    ("S63.501A", "Unspecified sprain of right wrist, initial encounter", "wrist-hand"),

    ("S73.101A", "Unspecified sprain of right hip, initial encounter", "hip"),
    ("S72.001A", "Fracture of unspecified part of neck of right femur, initial encounter", "hip"),

    ("S93.401A", "Sprain of unspecified ligament of right ankle, initial encounter", "ankle-foot"),
    ("S92.301A", "Unspecified fracture of right foot, initial encounter", "ankle-foot"),

    ("S06.0X0A", "Concussion without loss of consciousness, initial encounter", "head"),

    ("G56.00", "Carpal tunnel syndrome, unspecified upper limb", "wrist-hand"),

    ("S53.401A", "Unspecified sprain of right elbow, initial encounter", "elbow"),
    ("M77.10", "Lateral epicondylitis, unspecified elbow", "elbow"),
]

ICD10_HEADER = ["code", "description", "body_region"]


# ---------------------------------------------------------------------------
# NCCI Procedure-to-Procedure (PTP) edit pairs
# ---------------------------------------------------------------------------
# column1_code (the code that "survives"), column2_code (bundled into column1),
# modifier_indicator: 0 = never separately payable, 1 = separately payable
#   with an appropriate modifier (e.g. -59) if documentation supports it,
# rationale, source

NCCI_PTP_EDITS = [
    ("29827", "29826", "1",
     "Subacromial decompression is bundled into rotator cuff repair on the same shoulder, same session, unless documented as a distinct procedure.",
     "cms_documented"),
    ("29881", "29877", "1",
     "Chondroplasty is bundled into meniscectomy on the same knee compartment, same session.",
     "cms_documented"),
    ("29880", "29877", "1",
     "Chondroplasty is bundled into meniscectomy on the same knee, same session.",
     "cms_documented"),
    ("97110", "97140", "1",
     "Manual therapy and therapeutic exercise are bundled on the same date unless distinct regions/timed segments are documented.",
     "cms_documented"),
    ("97530", "97112", "1",
     "Therapeutic activities and neuromuscular reeducation are bundled on the same date unless distinct timed segments are documented.",
     "cms_documented"),

    ("27447", "29881", "0",
     "Arthroscopic meniscectomy is bundled into total knee arthroplasty performed on the same knee, same session.",
     "illustrative_pattern"),
    ("23472", "29826", "0",
     "Arthroscopic subacromial decompression is bundled into total shoulder arthroplasty on the same shoulder, same session.",
     "illustrative_pattern"),
    ("27130", "29862", "0",
     "Hip arthroscopy with labral repair is bundled into total hip arthroplasty on the same hip, same session.",
     "illustrative_pattern"),
    ("97116", "97110", "1",
     "Gait training and therapeutic exercise are bundled on the same date unless distinct timed segments are documented.",
     "illustrative_pattern"),
    ("20610", "20605", "1",
     "Major and intermediate joint injections on the same date are bundled unless performed on clinically distinct joints.",
     "illustrative_pattern"),
    ("25607", "20680", "1",
     "Deep implant removal is bundled into open reduction internal fixation performed at the same operative site, same session.",
     "illustrative_pattern"),
    ("29862", "20610", "1",
     "Major joint injection is bundled into hip arthroscopy with labral repair on the same hip, same session.",
     "illustrative_pattern"),
]

NCCI_PTP_HEADER = ["column1_code", "column2_code", "modifier_indicator", "rationale", "source"]


# ---------------------------------------------------------------------------
# Medically Unlikely Edits (max units per code per day)
# ---------------------------------------------------------------------------
# code, mue_units, rationale

_MUE_RULES = [
    # (codes, units, rationale)
    (["99202", "99203", "99204", "99205", "99212", "99213", "99214", "99215", "99221", "99231"],
     1, "E/M visit codes: one encounter of a given level per provider per day."),
    (["97110", "97112", "97116", "97140", "97530", "97124"],
     4, "15-minute timed therapy codes: CMS convention caps most at ~4 units (~1 hour) per day absent unusual documentation."),
    (["97535"],
     2, "Self-care/home management training: shorter typical session length than core PT/OT codes."),
    (["23472", "27130", "27447", "29826", "29827", "29877", "29880", "29881", "29862",
      "25607", "26720", "27758", "23600", "24500", "24505", "20670", "20680"],
     1, "Discrete surgical procedure: one occurrence per operative site per day."),
    (["20550"], 1, "Single tendon sheath/ligament injection site per day, absent multi-site documentation."),
    (["20605"], 2, "Intermediate joint injection: allows for bilateral joints in one day."),
    (["20610"], 2, "Major joint injection: allows for bilateral joints in one day."),
    (["29075", "29125", "29405"], 1, "One cast/splint application per limb per day."),
    (["72110", "73030", "73721", "72148"], 1, "One imaging study of a given type per day, absent re-imaging documentation."),
]

MUE_TABLE = [(code, units, rationale) for codes, units, rationale in _MUE_RULES for code in codes]
MUE_HEADER = ["code", "mue_units", "rationale"]


def write_csv(filename, header, rows):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {len(rows):>3} rows -> {path}")


def main():
    write_csv("cpt_hcpcs_codes.csv", CPT_HCPCS_HEADER, CPT_HCPCS_CODES)
    write_csv("icd10_codes.csv", ICD10_HEADER, ICD10_CODES)
    write_csv("ncci_ptp_edits.csv", NCCI_PTP_HEADER, NCCI_PTP_EDITS)
    write_csv("mue_table.csv", MUE_HEADER, MUE_TABLE)

    # sanity checks
    cpt_codes = {row[0] for row in CPT_HCPCS_CODES}
    for c1, c2, *_ in NCCI_PTP_EDITS:
        assert c1 in cpt_codes, f"NCCI column1 code {c1} not in CPT/HCPCS reference"
        assert c2 in cpt_codes, f"NCCI column2 code {c2} not in CPT/HCPCS reference"
    mue_codes = {row[0] for row in MUE_TABLE}
    missing_mue = cpt_codes - mue_codes
    if missing_mue:
        print(f"NOTE: {len(missing_mue)} CPT/HCPCS codes have no MUE row: {sorted(missing_mue)}")
    print("reference data build OK")


if __name__ == "__main__":
    main()
