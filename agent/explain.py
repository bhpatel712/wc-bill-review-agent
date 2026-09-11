"""
explain.py  (WK4.3)

Turns a retrieved rule document + the specific facts of a flagged bill
into a plain-language explanation, e.g. "codes X and Y were billed
together, but NCCI edits indicate Y is bundled into X for this service
date" (the exact example from the WK4.3 plan).

The model is given the retrieved rule text and the bill's specific facts
directly in the prompt and told explicitly not to rely on anything else --
this is the actual point of doing retrieval before generation: the model
isn't asked to recall workers' comp coding rules from its own training
data (which it might get wrong, or which might be stale relative to this
project's reference tables), it's handed the specific rule this project
already verified applies and just asked to explain it in plain language.
That's what keeps the explanation grounded instead of the model quietly
substituting a plausible-sounding rule it half-remembers.

Run:
    python3 explain.py
"""
import json
import os
import sys

from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_chat_model  # noqa: E402
from retrieve import get_vectorstore, load_reference_for_checking, retrieve_rule, BILLS_PATH  # noqa: E402

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You explain workers' compensation medical billing anomalies to a claims "
     "reviewer in plain, direct English. You are given the SPECIFIC rule that "
     "was violated and the SPECIFIC facts of the bill -- use only those, don't "
     "add outside knowledge or invent details that aren't provided. "
     "Write 2-3 sentences: state what was billed, cite the specific rule that "
     "makes it a problem, and say plainly why that matters. No preamble, no "
     "headers, no bullet points -- just the explanation."),
    ("human",
     "Bill {bill_id}, provider specialty: {provider_specialty}.\n"
     "Violation type: {violation_type}\n"
     "Bill-specific facts: {bill_facts}\n"
     "Retrieved rule: {rule_text}\n\n"
     "Write the plain-language explanation."),
])


def bill_facts_for(bill, violation_type, detail):
    if violation_type == "unbundling":
        return f"Billed CPT {detail['col1']} and CPT {detail['col2']} on the same date of service ({bill['date_of_service']})."
    if violation_type == "fragmented_billing":
        return f"Billed {detail['units']} units of CPT {detail['cpt_code']} on {bill['date_of_service']}."
    if violation_type == "causality_mismatch":
        diag = ", ".join(sorted(detail["diag_regions"]))
        proc = ", ".join(sorted(detail["proc_regions"])) or "none region-specific"
        return f"Diagnosis region(s): {diag}. Procedure region(s) billed: {proc}."
    if violation_type == "upcoding":
        return f"Billed CPT {detail['cpt_code']} while synthetic documentation level is {bill['documentation_level']}/5."
    return ""


def explain_bill(bill, violation_type, vectorstore, ncci_edits, mue, icd_region_lookup, chat_model):
    doc, detail = retrieve_rule(bill, violation_type, vectorstore, ncci_edits, mue, icd_region_lookup)
    if doc is None:
        return None

    chain = PROMPT | chat_model
    result = chain.invoke({
        "bill_id": bill["bill_id"],
        "provider_specialty": bill["provider_specialty"],
        "violation_type": violation_type,
        "bill_facts": bill_facts_for(bill, violation_type, detail),
        "rule_text": doc.page_content,
    })
    return result.content if hasattr(result, "content") else str(result)


def demo():
    ncci_edits, mue, icd_region_lookup = load_reference_for_checking()
    vectorstore = get_vectorstore()
    chat_model = get_chat_model()

    with open(BILLS_PATH) as f:
        bills = [json.loads(line) for line in f]

    seen_types = set()
    for bill in bills:
        for vt in bill["anomaly_types"]:
            if vt in seen_types:
                continue
            explanation = explain_bill(bill, vt, vectorstore, ncci_edits, mue, icd_region_lookup, chat_model)
            print(f"\n=== {bill['bill_id']}  ({vt}) ===")
            print("ground truth (from generate_bills.py):", bill["anomaly_details"][0])
            print("generated explanation:")
            print(" ", explanation)
            seen_types.add(vt)
        if seen_types == {"unbundling", "fragmented_billing", "causality_mismatch", "upcoding"}:
            break


if __name__ == "__main__":
    demo()
