"""
retrieve.py  (WK4.2)

Given a flagged bill, finds the specific rule document it violated.

Two different retrieval strategies, deliberately -- not everything here
should go through fuzzy semantic search just because this is a "RAG"
project:

  - unbundling / fragmented_billing: the exact NCCI code pair or MUE-capped
    code involved is a fact we can compute directly from the bill's lines
    (reusing generate_bills.py's own ground-truth checker, so this uses
    the exact same detection logic the corpus was labeled with). Once we
    know the exact code(s), we retrieve by an exact metadata filter on the
    vector store rather than hoping a similarity search lands on the right
    document -- more reliable, and just as much "using the vector store"
    as a semantic query is.

  - causality_mismatch / upcoding: there's no per-code reference row to
    look up (see rule_docs.py) -- these hand-written concept documents are
    what semantic search is actually for here, matching the bill's
    specific situation (which body regions, which E/M gap) against the
    general principle that best explains it.

Run:
    python3 retrieve.py                  # demo: one example bill per violation type
"""
import json
import os
import sys

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data-generation"))
from generate_bills import violates_any_rule  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(HERE, "chroma_db")
COLLECTION_NAME = "wc_bill_review_rules"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

REF_DIR = os.path.join(HERE, "..", "data-generation", "reference")
BILLS_PATH = os.path.join(HERE, "..", "data-generation", "output", "synthetic_bills.jsonl")


def load_reference_for_checking():
    import csv
    with open(os.path.join(REF_DIR, "ncci_ptp_edits.csv")) as f:
        ncci_edits = [
            {"col1": r["column1_code"], "col2": r["column2_code"],
             "modifier_indicator": r["modifier_indicator"], "rationale": r["rationale"], "source": r["source"]}
            for r in csv.DictReader(f)
        ]
    # violates_any_rule() (generate_bills.py) expects mue[code] = {"units": ..., "rationale": ...},
    # matching the shape its own load_reference() builds from the SQLite table --
    # NOT the flat mue[code] = int shape features.py's load_mue() uses for a
    # different purpose. Mixing the two up is a real bug this project already
    # hit once (caught by a mechanical test before this ever ran for real).
    mue = {}
    with open(os.path.join(REF_DIR, "mue_table.csv")) as f:
        for r in csv.DictReader(f):
            mue[r["code"]] = {"units": int(r["mue_units"]), "rationale": r["rationale"]}
    icd_region_lookup = {}
    with open(os.path.join(REF_DIR, "icd10_codes.csv")) as f:
        for r in csv.DictReader(f):
            icd_region_lookup[r["code"]] = r["body_region"]
    return ncci_edits, mue, icd_region_lookup


def get_vectorstore():
    if not os.path.isdir(PERSIST_DIR):
        raise RuntimeError(f"No index found at {PERSIST_DIR} -- run build_index.py first.")
    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=PERSIST_DIR)


def retrieve_rule(bill, violation_type, vectorstore, ncci_edits, mue, icd_region_lookup):
    """Returns (retrieved_document, detail) -- detail is the ground-truth
    fact (an NCCI edit dict, an MUE-exceeding line, region sets, or an
    upcoded line) that violates_any_rule found, kept alongside the
    retrieved document so explain.py has both the specific bill facts and
    the general rule text to work from."""
    hits = violates_any_rule(bill, ncci_edits, mue, icd_region_lookup)
    detail = next((d for vt, d in hits if vt == violation_type), None)
    if detail is None:
        return None, None

    if violation_type == "unbundling":
        results = vectorstore.similarity_search(
            "NCCI unbundling edit", k=1,
            filter={"$and": [{"code1": detail["col1"]}, {"code2": detail["col2"]}]},
        )
    elif violation_type == "fragmented_billing":
        results = vectorstore.similarity_search(
            "MUE unit cap", k=1,
            filter={"$and": [{"rule_type": "fragmented_billing"}, {"code": detail["cpt_code"]}]},
        )
    elif violation_type == "causality_mismatch":
        proc_regions = ", ".join(sorted(detail["proc_regions"])) or "none"
        diag_regions = ", ".join(sorted(detail["diag_regions"]))
        query = f"diagnosis region {diag_regions} does not match procedure region {proc_regions} medical necessity"
        results = vectorstore.similarity_search(query, k=1, filter={"rule_type": "causality_mismatch"})
    elif violation_type == "upcoding":
        query = f"E/M code {detail['cpt_code']} billed above documentation level {bill['documentation_level']}"
        results = vectorstore.similarity_search(query, k=1, filter={"rule_type": "upcoding"})
    else:
        results = []

    return (results[0] if results else None), detail


def demo():
    ncci_edits, mue, icd_region_lookup = load_reference_for_checking()
    vectorstore = get_vectorstore()

    with open(BILLS_PATH) as f:
        bills = [json.loads(line) for line in f]

    seen_types = set()
    for bill in bills:
        for vt in bill["anomaly_types"]:
            if vt in seen_types:
                continue
            doc, detail = retrieve_rule(bill, vt, vectorstore, ncci_edits, mue, icd_region_lookup)
            print(f"\n=== {bill['bill_id']}  ({vt}) ===")
            print("ground truth detail from generate_bills.py:", bill["anomaly_details"][0])
            print("retrieved rule document:")
            print(" ", doc.page_content if doc else "(none found)")
            print(" metadata:", doc.metadata if doc else None)
            seen_types.add(vt)
        if seen_types == {"unbundling", "fragmented_billing", "causality_mismatch", "upcoding"}:
            break


if __name__ == "__main__":
    demo()
