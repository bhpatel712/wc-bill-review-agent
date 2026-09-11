"""
schemas.py  (WK6 support)

Pydantic request/response models for the FastAPI service. BillIn mirrors
the *input* fields of a synthetic_bills.jsonl record exactly (see
data-generation/DATA_CARD.md) -- everything except the ground-truth
labels (is_anomalous, anomaly_types, anomaly_details), which a real
incoming bill wouldn't have. This is deliberately the same shape the
Week 1 generator produces, so any bill from the synthetic corpus can be
posted to this API unmodified (just strip the label fields) as a smoke
test.
"""
from typing import Optional

from pydantic import BaseModel, Field

VIOLATION_TYPES = ["unbundling", "fragmented_billing", "causality_mismatch", "upcoding"]


class BillLine(BaseModel):
    cpt_code: str
    units: int = Field(ge=1)
    modifier: str = ""
    charge: float = Field(ge=0)


class BillIn(BaseModel):
    bill_id: str
    provider_id: str
    provider_specialty: str
    injury_body_region: str
    date_of_service: str
    days_since_injury: int = Field(ge=0)
    injury_stage: str
    is_new_patient: bool
    documentation_level: int = Field(ge=1, le=5)
    diagnosis_codes: list[str]
    lines: list[BillLine]
    total_billed: float = Field(ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "bill_id": "WC-BILL-000066",
                "provider_id": "WC-PR-0006",
                "provider_specialty": "Occupational Medicine",
                "injury_body_region": "shoulder",
                "date_of_service": "2025-11-16",
                "days_since_injury": 89,
                "injury_stage": "late",
                "is_new_patient": False,
                "documentation_level": 5,
                "diagnosis_codes": ["S43.401A"],
                "lines": [
                    {"cpt_code": "99215", "units": 1, "modifier": "", "charge": 235.56},
                    {"cpt_code": "97110", "units": 3, "modifier": "", "charge": 186.45},
                    {"cpt_code": "97112", "units": 2, "modifier": "", "charge": 113.65},
                ],
                "total_billed": 535.66,
            }
        }


class LabelPrediction(BaseModel):
    probability: float
    flagged: bool
    # Which CPT code(s) on THIS bill are the ones actually implicated in
    # this violation type -- e.g. the two codes in an NCCI pair, the one
    # code over its MUE cap, the one upcoded E/M line, or (for
    # causality_mismatch) every procedure code billed in a body region
    # that doesn't match the diagnosis. Computed for free, locally, by
    # code_lookup.py's deterministic checker -- no LLM call, no cost.
    # Empty for a non-flagged type, and also empty if the model flagged
    # this type but the deterministic checker didn't confirm it (a likely
    # model false positive on this specific bill).
    involved_codes: list[str] = []


class PredictResponse(BaseModel):
    bill_id: str
    predictions: dict[str, LabelPrediction]
    any_flagged: bool


class ExplainRequest(BaseModel):
    bill: BillIn
    violation_type: str


class ExplainResponse(BaseModel):
    bill_id: str
    violation_type: str
    rule_detail_found: bool
    explanation: Optional[str] = None
    rule_document: Optional[str] = None
    note: Optional[str] = None


class ReviewResponse(BaseModel):
    bill_id: str
    predictions: dict[str, LabelPrediction]
    any_flagged: bool
    explanations: Optional[dict[str, ExplainResponse]] = None


class BillLookupResponse(BaseModel):
    bill: BillIn
    # Ground truth from the synthetic corpus's own generator (Week 1) --
    # only exists because this is a demo bill from a labeled synthetic
    # dataset. A bill submitted from outside the corpus has no such label;
    # this field exists purely so the UI can show "here's what the model
    # said vs. what this bill was actually constructed to be" for bills
    # that DO come from the corpus, as an honesty/trust check on the demo
    # itself -- never present for a real incoming bill in a real system.
    ground_truth_is_anomalous: bool
    ground_truth_anomaly_types: list[str]
