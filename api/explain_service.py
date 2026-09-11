"""
explain_service.py  (WK6 support)

Thin wrapper around the Week 4 RAG agent (agent/retrieve.py,
agent/explain.py) for the API layer.

Deliberately lazy: importing this module does NOT load the Chroma vector
store or build an Azure OpenAI client -- those only happen the first time
get_service() actually runs. That means /predict and /health work fine
even when agent/.env isn't configured or agent/chroma_db hasn't been
built yet; only /explain (and /review?generate_explanations=true) need
this, and they fail with one clear message instead of the whole process
refusing to start.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(HERE, "..", "agent")
sys.path.insert(0, AGENT_DIR)


class ExplainNotReadyError(RuntimeError):
    """Raised when the RAG explainer can't be used yet -- missing
    agent/.env values, agent/chroma_db not built, or agent/'s own
    dependencies (langchain, etc.) not installed."""


class ExplainService:
    def __init__(self):
        # Imported here, not at module load time, specifically so a
        # missing chroma_db / unset .env doesn't break *importing* this
        # module (and therefore api/main.py) -- only constructing this
        # class does, and only when /explain is actually called.
        try:
            from config import get_chat_model, get_missing_vars
            from retrieve import get_vectorstore, load_reference_for_checking, retrieve_rule
            from explain import PROMPT, bill_facts_for
        except ImportError as e:
            raise ExplainNotReadyError(f"agent/ dependencies not installed: {e}") from e

        missing = get_missing_vars()
        if missing:
            raise ExplainNotReadyError(
                "Azure OpenAI not configured -- missing " + ", ".join(missing) +
                " in agent/.env. See agent/RAG_AGENT.md's Setup section."
            )

        try:
            self.vectorstore = get_vectorstore()
        except RuntimeError as e:
            raise ExplainNotReadyError(f"{e} Run `python3 agent/build_index.py` first.") from e

        self.ncci_edits, self.mue, self.icd_region_lookup = load_reference_for_checking()
        self.chat_model = get_chat_model()
        self._retrieve_rule = retrieve_rule
        self._prompt = PROMPT
        self._bill_facts_for = bill_facts_for

    def explain(self, bill: dict, violation_type: str) -> dict:
        doc, detail = self._retrieve_rule(
            bill, violation_type, self.vectorstore,
            self.ncci_edits, self.mue, self.icd_region_lookup,
        )
        if doc is None:
            # This bill doesn't actually trip this violation type under
            # the same deterministic checker the training corpus was
            # labeled with (generate_bills.py's violates_any_rule) -- so
            # there's no ground-truth rule text to explain. If this call
            # came from /review's auto-explain of a model-flagged type,
            # that combination (model says yes, deterministic checker
            # says no) is itself a meaningful signal: a likely model
            # false positive on this specific bill, surfaced honestly
            # instead of generating an explanation for something that
            # didn't actually happen.
            return {
                "rule_detail_found": False,
                "explanation": None,
                "rule_document": None,
                "note": (
                    f"No ground-truth '{violation_type}' violation found for this bill "
                    "by the same rule-based checker the training corpus was labeled with. "
                    "If a model prediction flagged this type, this is most likely a model "
                    "false positive on this specific bill -- there's nothing to ground an "
                    "explanation in, so none was generated."
                ),
            }

        chain = self._prompt | self.chat_model
        result = chain.invoke({
            "bill_id": bill["bill_id"],
            "provider_specialty": bill["provider_specialty"],
            "violation_type": violation_type,
            "bill_facts": self._bill_facts_for(bill, violation_type, detail),
            "rule_text": doc.page_content,
        })
        explanation = result.content if hasattr(result, "content") else str(result)
        return {
            "rule_detail_found": True,
            "explanation": explanation,
            "rule_document": doc.page_content,
            "note": None,
        }


_service = None


def get_service() -> ExplainService:
    """Builds the service on first call and reuses it after -- the vector
    store and chat model are expensive enough to set up once per process,
    not once per request."""
    global _service
    if _service is None:
        _service = ExplainService()
    return _service
