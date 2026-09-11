"""
rule_docs.py  (WK4.1 support)

Builds the LangChain Document objects that get embedded and indexed by
build_index.py. Two kinds of source material:

  1. Per-row documents from the real reference tables built in Week 1
     (data-generation/reference/ncci_ptp_edits.csv, mue_table.csv) -- one
     document per NCCI edit pair, one per MUE-capped code. These have an
     exact code (or code pair) attached as metadata, so retrieve.py can
     look them up with a metadata filter instead of guessing from text.

  2. Hand-written "concept" documents for the two violation types that
     have no per-code reference table at all: causality_mismatch (there's
     no CMS table of "which diagnoses go with which procedures" in this
     project's reference data -- see DATA_CARD.md) and upcoding (E/M level
     selection is a documentation-support judgment, not a per-code CMS
     table either). These are grounded in real, well-known coding
     principles -- medical necessity, and E/M documentation-supports-level
     -- written in our own words, not copied from any licensed manual.

Every Document's metadata includes "rule_type" (one of the four violation
types) so retrieve.py can filter the vector store down to only the
documents that are even possibly relevant before it searches.
"""
import csv
import os

from langchain_core.documents import Document

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "..", "data-generation", "reference")


def load_ncci_docs():
    docs = []
    with open(os.path.join(REF_DIR, "ncci_ptp_edits.csv")) as f:
        for row in csv.DictReader(f):
            col1, col2 = row["column1_code"], row["column2_code"]
            bypassable = "yes, with modifier 59 if truly a distinct procedure" if row["modifier_indicator"] == "1" else "no -- never separately payable, regardless of modifier"
            text = (
                f"NCCI PTP edit: CPT {col1} and CPT {col2} billed on the same date of service. "
                f"Separately payable with a bypass modifier? {bypassable}. "
                f"{row['rationale']} (source: {row['source']})"
            )
            docs.append(Document(
                page_content=text,
                metadata={
                    "rule_type": "unbundling",
                    "code1": col1,
                    "code2": col2,
                    "modifier_indicator": row["modifier_indicator"],
                    "source": row["source"],
                },
            ))
    return docs


def load_mue_docs():
    docs = []
    with open(os.path.join(REF_DIR, "mue_table.csv")) as f:
        for row in csv.DictReader(f):
            text = (
                f"MUE (Medically Unlikely Edit) rule: CPT/HCPCS {row['code']} -- "
                f"CMS daily unit cap is {row['mue_units']} unit(s) per provider per date of service. "
                f"{row['rationale']}"
            )
            docs.append(Document(
                page_content=text,
                metadata={
                    "rule_type": "fragmented_billing",
                    "code": row["code"],
                    "mue_units": int(row["mue_units"]),
                },
            ))
    return docs


def causality_mismatch_concept_docs():
    return [
        Document(
            page_content=(
                "Medical necessity: a billed procedure must be reasonable and necessary for "
                "the diagnosis it's billed against. A payer expects the procedure and the "
                "diagnosis to describe the same clinical problem -- e.g. physical therapy "
                "codes billed against a lumbar spine diagnosis, or a knee arthroscopy billed "
                "against a knee diagnosis. When the diagnosis on a bill describes one body "
                "region and every procedure billed is specific to a different, unrelated body "
                "region, there is no plausible clinical link between what was diagnosed and "
                "what was treated -- a causality mismatch."
            ),
            metadata={"rule_type": "causality_mismatch", "topic": "medical_necessity"},
        ),
        Document(
            page_content=(
                "Diagnosis-procedure body region consistency: in workers' compensation billing "
                "specifically, treatment must trace back to the compensable injury. A diagnosis "
                "coded to one body region (e.g. shoulder) paired with procedures that only make "
                "sense for a different body region (e.g. ankle/foot codes) is a red flag for "
                "either a coding error (wrong diagnosis or wrong procedure code entered) or "
                "billing for treatment unrelated to the accepted injury -- both of which a payer "
                "or auditor would query before reimbursing."
            ),
            metadata={"rule_type": "causality_mismatch", "topic": "region_consistency"},
        ),
    ]


def upcoding_concept_docs():
    return [
        Document(
            page_content=(
                "E/M (Evaluation and Management) code selection must be supported by the "
                "documented complexity of the visit -- the history, exam, and medical "
                "decision-making (or, under time-based rules, total time) actually recorded in "
                "the note. E/M levels run 1 (straightforward) through 5 (highest complexity); "
                "billing a higher-level E/M code than the documentation supports -- for example "
                "billing the highest-complexity code against a note that only documents a brief, "
                "low-complexity encounter -- is upcoding, regardless of intent."
            ),
            metadata={"rule_type": "upcoding", "topic": "em_documentation_support"},
        ),
        Document(
            page_content=(
                "'Code to the documentation, not to the payer' is a standard coding-compliance "
                "principle: the billed code must reflect what the clinical note actually "
                "supports, not the highest code that would be reimbursed at the best rate. A "
                "gap between the E/M level billed and the documentation level actually recorded "
                "is exactly the pattern a compliance audit looks for when screening for upcoding."
            ),
            metadata={"rule_type": "upcoding", "topic": "code_to_documentation"},
        ),
    ]


def build_all_docs():
    return (
        load_ncci_docs()
        + load_mue_docs()
        + causality_mismatch_concept_docs()
        + upcoding_concept_docs()
    )


if __name__ == "__main__":
    docs = build_all_docs()
    by_type = {}
    for d in docs:
        by_type[d.metadata["rule_type"]] = by_type.get(d.metadata["rule_type"], 0) + 1
    print(f"built {len(docs)} rule documents: {by_type}")
    print("\nsample document:")
    print(docs[0].page_content)
    print(docs[0].metadata)
