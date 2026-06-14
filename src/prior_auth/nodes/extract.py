from langchain_core.messages import SystemMessage

from prior_auth.llm import get_llm
from prior_auth.prompts.extract import EXTRACTION_PROMPT, EXTRACTION_RETRY_HINT
from prior_auth.schemas.extraction import ExtractedRequest
from prior_auth.state import PriorAuthState

# Below this self-reported confidence, flag the case for human review.
LOW_CONFIDENCE_THRESHOLD = 0.6


def extract_request(state: PriorAuthState) -> dict:
    """Parse the raw referral note into structured fields.

    Always returns a partial state update with `extracted_request` and
    `extraction_flags` -- never raises. An unparseable request is itself
    a "needs human review" case, not a system error.
    """
    structured_llm = get_llm().with_structured_output(ExtractedRequest)
    messages = EXTRACTION_PROMPT.format_messages(raw_text=state["raw_request_text"])

    try:
        extracted = structured_llm.invoke(messages)
    except Exception:
        # with_structured_output can fail in a few different ways (schema
        # validation, missing tool call, etc.) -- one retry with a more
        # directive prompt before falling back to a flagged placeholder.
        retry_messages = messages + [SystemMessage(content=EXTRACTION_RETRY_HINT)]
        try:
            extracted = structured_llm.invoke(retry_messages)
        except Exception:
            extracted = ExtractedRequest(
                procedure_description="EXTRACTION_FAILED",
                diagnosis_description="EXTRACTION_FAILED",
                clinical_justification="EXTRACTION_FAILED",
                extraction_confidence=0.0,
            )
            return {
                "extracted_request": extracted,
                "extraction_flags": ["extraction_error"],
            }

    return {"extracted_request": extracted, "extraction_flags": _check_flags(extracted)}


def _check_flags(extracted: ExtractedRequest) -> list[str]:
    """Heuristic checks layered on top of the LLM's self-reported
    confidence -- catch missing fields the model itself might not flag."""
    flags = []
    if extracted.extraction_confidence < LOW_CONFIDENCE_THRESHOLD:
        flags.append("low_confidence_extraction")
    if extracted.diagnosis_code is None:
        flags.append("missing_diagnosis_code")
    if extracted.procedure_code is None:
        flags.append("missing_procedure_code")
    return flags
