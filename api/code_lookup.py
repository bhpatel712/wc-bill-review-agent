"""
code_lookup.py  (WK6 support)

Free, local lookup of which CPT code(s) on a bill are the ones actually
implicated in a flagged violation type -- reuses the exact same
deterministic ground-truth checker the training corpus was labeled with
(data-generation/generate_bills.py's violates_any_rule()), NOT the LLM.
No Azure OpenAI call, no vector store -- this is pure bill-lines lookup
against the same reference CSVs (NCCI PTP edits, MUE table, ICD-10 region
lookup) agent/retrieve.py already uses -- so it's safe and free to run on
every /predict and /review response, regardless of whether "Generate
explanations" is toggled on.

Deliberately reuses agent/retrieve.py's load_reference_for_checking()
rather than models/features.py's load_mue()/load_ncci_pairs() -- those
build DIFFERENT shapes for a different purpose (see retrieve.py's own
comment about this exact mix-up), and violates_any_rule() needs the
shape load_reference_for_checking() builds.

Loaded eagerly at API startup (like ModelService in inference.py), not
lazily (like ExplainService) -- it's free and local, same reasoning
main.py already documents for the model vs. the RAG explainer.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(HERE, "..", "agent")
DATA_GEN_DIR = os.path.join(HERE, "..", "data-generation")
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, DATA_GEN_DIR)

from retrieve import load_reference_for_checking  # noqa: E402  (agent/retrieve.py -- CSV loading only, no vectorstore/Azure)
from generate_bills import violates_any_rule, REGION_CPT_POOL  # noqa: E402  (data-generation/generate_bills.py)


class CodeLookupNotReadyError(RuntimeError):
    """Raised when the reference CSVs (NCCI/MUE/ICD-region tables) aren't
    available yet -- the same Week 1 reference files models/inference.py
    already depends on, just loaded in the shape violates_any_rule()
    expects rather than the shape features.py builds."""


class CodeLookupService:
    def __init__(self):
        try:
            self.ncci_edits, self.mue, self.icd_region_lookup = load_reference_for_checking()
        except FileNotFoundError as e:
            raise CodeLookupNotReadyError(
                f"Missing a required reference file ({e.filename}). Run the Week 1 data "
                "generation step first (see README.md's \"Running it end to end\")."
            ) from e

    def involved_codes_by_type(self, bill: dict) -> dict:
        """Runs the deterministic checker once and buckets the implicated
        CPT code(s) per violation type it actually hit on THIS bill.
        A type absent from the checker's hits (e.g. the model flagged it
        but the ground-truth checker didn't -- a likely model false
        positive, the same case explain_service.py already surfaces
        honestly) simply gets an empty list, not an error."""
        hits = violates_any_rule(bill, self.ncci_edits, self.mue, self.icd_region_lookup)
        result = {"unbundling": [], "fragmented_billing": [], "causality_mismatch": [], "upcoding": []}

        for violation_type, detail in hits:
            if violation_type == "unbundling":
                codes = [detail["col1"], detail["col2"]]
            elif violation_type in ("fragmented_billing", "upcoding"):
                codes = [detail["cpt_code"]]
            elif violation_type == "causality_mismatch":
                # Not one bad code -- every procedure code billed in a body
                # region that doesn't match any billed diagnosis code.
                proc_regions = detail["proc_regions"]
                codes = []
                for l in bill["lines"]:
                    for region, pool in REGION_CPT_POOL.items():
                        if region in proc_regions and l["cpt_code"] in pool and l["cpt_code"] not in codes:
                            codes.append(l["cpt_code"])
                            break
            else:
                codes = []
            # A violation type normally appears once per bill in
            # violates_any_rule()'s output, except fragmented_billing, which
            # appends one hit per over-cap line -- so accumulate rather than
            # overwrite, preserving order and dropping duplicates.
            for code in codes:
                if code not in result[violation_type]:
                    result[violation_type].append(code)

        return result


_service = None
_service_error = None


def get_service():
    """Builds the service on first call and reuses it after. Unlike
    explain_service.get_service(), failure here is cached too (not
    retried every request) since it only depends on local files that
    won't start existing mid-process."""
    global _service, _service_error
    if _service is None and _service_error is None:
        try:
            _service = CodeLookupService()
        except CodeLookupNotReadyError as e:
            _service_error = e
    if _service_error is not None:
        raise _service_error
    return _service
