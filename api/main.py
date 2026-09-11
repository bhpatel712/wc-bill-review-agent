"""
main.py  (WK6.1-WK6.3)

FastAPI service tying the whole project together behind HTTP endpoints:

  GET  /health   -- liveness + readiness: is the trained model loaded,
                     is the RAG explainer's config present. Never itself
                     calls Azure OpenAI (that would cost money on every
                     health poll) -- only checks whether it COULD.
  POST /predict   -- bill -> per-violation-type probability + flag, from
                     the Week 2 XGBoost model. Local inference only, no
                     external calls, no cost.
  POST /explain   -- bill + one violation_type -> a grounded plain-
                     language explanation via the Week 4 RAG agent.
                     Calls Azure OpenAI -- real, small, per-call cost.
  POST /review    -- convenience endpoint: /predict, and (only if you opt
                     in with ?generate_explanations=true) /explain for
                     every flagged type. Off by default on purpose: this
                     is the only place in this API with a real cost, and
                     it shouldn't be possible to rack one up by accident.

Run (from the project root, after Weeks 1-4's pipeline has been run once):
    uvicorn api.main:app --reload
Then see api/API.md for curl examples, or open http://127.0.0.1:8000/docs
for FastAPI's interactive docs.
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Make this file's own directory (api/) importable as top-level modules
# (code_lookup, schemas, inference, explain_service below), regardless of
# how uvicorn is launched. Needed because "uvicorn api.main:app --reload"
# (the documented run command, invoked from the project root) only puts
# the project root on sys.path -- not api/ itself -- so these bare
# same-directory imports would otherwise raise ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from code_lookup import CodeLookupNotReadyError, get_service as get_code_lookup_service
from explain_service import ExplainNotReadyError, get_service as get_explain_service
from inference import ModelNotReadyError, ModelService
from schemas import (
    VIOLATION_TYPES,
    BillIn,
    BillLookupResponse,
    ExplainRequest,
    ExplainResponse,
    LabelPrediction,
    PredictResponse,
    ReviewResponse,
)

# explain_service.py already put agent/ on sys.path when imported above;
# reusing that here for a cheap (no network) config presence check in
# /health, rather than duplicating the path-setup logic.
from config import get_missing_vars  # noqa: E402  (agent/config.py)

model_service: ModelService | None = None
model_error: str | None = None
code_lookup_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_service, model_error, code_lookup_error
    # The trained model + its supporting reference data are local, free,
    # and needed by the API's core purpose -- load them eagerly at
    # startup. The RAG explainer is the opposite (needs credentials, has
    # a real per-call cost) and is deliberately NOT loaded here -- see
    # explain_service.get_service()'s lazy init.
    try:
        model_service = ModelService()
    except ModelNotReadyError as e:
        model_error = str(e)

    # The CPT-code lookup (code_lookup.py) is local and free too -- same
    # reasoning, load it eagerly. A failure here (missing reference CSVs)
    # shouldn't take down the whole API -- predictions still work, they
    # just won't carry involved_codes -- so this is caught, not raised.
    try:
        get_code_lookup_service()
    except CodeLookupNotReadyError as e:
        code_lookup_error = str(e)
    yield


app = FastAPI(
    title="WC Bill Review & Code Anomaly Detection API",
    description=(
        "Portfolio project API -- flags likely coding problems on 100% "
        "synthetic workers' comp bills and (opt-in) explains them. "
        "See the project README and api/API.md."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _require_model() -> ModelService:
    if model_service is None:
        raise HTTPException(status_code=503, detail=model_error or "Model not loaded.")
    return model_service


def _predictions_and_flags(svc: ModelService, bill: BillIn):
    raw = svc.predict(bill.model_dump())
    predictions = {k: LabelPrediction(**v) for k, v in raw.items()}

    # Attach the free, local "which CPT codes actually caused this" lookup.
    # Best-effort: if code_lookup.py's reference CSVs aren't available for
    # some reason, predictions still come back fine, just without codes --
    # this is a nice-to-have on top of /predict's core job, not a
    # dependency of it.
    try:
        involved = get_code_lookup_service().involved_codes_by_type(bill.model_dump())
        for violation_type, pred in predictions.items():
            pred.involved_codes = involved.get(violation_type, [])
    except CodeLookupNotReadyError:
        pass

    any_flagged = any(p.flagged for p in predictions.values())
    return predictions, any_flagged


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def ui():
    """Serves the chat-styled bill-review UI (api/static/index.html). It's a
    static page: no templating, no server-rendered state -- everything it
    needs (example bills, /review calls) happens client-side in the
    browser. API docs live separately at /docs."""
    return (STATIC_DIR / "index.html").read_text()


@app.get("/health")
def health():
    rag_env_ready = True
    rag_note = None
    try:
        missing = get_missing_vars()
        if missing:
            rag_env_ready = False
            rag_note = "agent/.env missing: " + ", ".join(missing)
    except Exception as e:  # config.py itself failing to import, etc.
        rag_env_ready = False
        rag_note = str(e)

    return {
        "status": "ok" if model_service is not None else "degraded",
        "model_loaded": model_service is not None,
        "model_error": model_error,
        # "configured", not "working" -- this only checks that the .env
        # values are present, not that they're valid or that Azure OpenAI
        # is reachable. That check is deferred to an actual /explain call,
        # deliberately, so /health never spends money to answer.
        "rag_env_configured": rag_env_ready,
        "rag_note": rag_note,
        # Whether /predict and /review can attach involved_codes (free,
        # local -- unrelated to the RAG explainer's Azure/.env readiness).
        "code_lookup_ready": code_lookup_error is None,
        "code_lookup_note": code_lookup_error,
    }


@app.get("/bills/{bill_id}", response_model=BillLookupResponse)
def get_bill(bill_id: str):
    """Looks up a bill by ID in the synthetic corpus -- backs the UI's
    "enter a bill ID" flow so a demo doesn't require hand-crafting bill
    JSON. Only finds bills that exist in data-generation/output/
    synthetic_bills.jsonl (i.e. ones the Week 1 generator produced); a
    real deployment wouldn't have this endpoint at all, since a real
    system receives whole bills, not IDs to look up from its own
    training data -- this exists purely to make the demo usable."""
    svc = _require_model()
    bill = svc.bills_by_id.get(bill_id)
    if bill is None:
        raise HTTPException(status_code=404, detail=f"No bill with id '{bill_id}' found in the synthetic corpus.")
    return BillLookupResponse(
        bill=BillIn(**bill),
        ground_truth_is_anomalous=bill["is_anomalous"],
        ground_truth_anomaly_types=bill["anomaly_types"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(bill: BillIn):
    svc = _require_model()
    predictions, any_flagged = _predictions_and_flags(svc, bill)
    return PredictResponse(bill_id=bill.bill_id, predictions=predictions, any_flagged=any_flagged)


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    if req.violation_type not in VIOLATION_TYPES:
        raise HTTPException(status_code=400, detail=f"violation_type must be one of {VIOLATION_TYPES}")
    try:
        svc = get_explain_service()
    except ExplainNotReadyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    result = svc.explain(req.bill.model_dump(), req.violation_type)
    return ExplainResponse(bill_id=req.bill.bill_id, violation_type=req.violation_type, **result)


@app.post("/review", response_model=ReviewResponse)
def review(
    bill: BillIn,
    generate_explanations: bool = Query(
        False,
        description=(
            "If true, also calls the Week 4 RAG agent (Azure OpenAI) to explain every "
            "flagged violation type. Off by default -- this is the only parameter in "
            "this whole API that can incur real cost, so it has to be asked for."
        ),
    ),
):
    svc = _require_model()
    predictions, any_flagged = _predictions_and_flags(svc, bill)

    explanations = None
    if generate_explanations:
        flagged_types = [vt for vt, p in predictions.items() if p.flagged]
        if flagged_types:
            try:
                explain_svc = get_explain_service()
            except ExplainNotReadyError as e:
                raise HTTPException(status_code=503, detail=str(e))
            explanations = {}
            for vt in flagged_types:
                result = explain_svc.explain(bill.model_dump(), vt)
                explanations[vt] = ExplainResponse(bill_id=bill.bill_id, violation_type=vt, **result)

    return ReviewResponse(
        bill_id=bill.bill_id,
        predictions=predictions,
        any_flagged=any_flagged,
        explanations=explanations,
    )
