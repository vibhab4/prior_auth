from langchain_core.messages import SystemMessage

from prior_auth.llm import get_llm
from prior_auth.prompts.propose import (
    PROPOSER_PROMPT,
    PROPOSER_RETRY_HINT,
    format_flags_section,
    format_policy_section,
    format_request_section,
)
from prior_auth.schemas.decision import ProposerDecision
from prior_auth.state import PriorAuthState


def propose_coverage(state: PriorAuthState) -> dict:
    """Evaluate the extracted request against retrieved policy excerpts.

    First LLM reasoning step in the pipeline. Produces an initial coverage
    recommendation (approve/deny/need_more_info) with step-by-step rationale
    and direct policy citations. This is a *proposal*, not the final decision
    -- the critic (Node 4) will challenge it.

    Reads: extracted_request, retrieved_policy_chunks, extraction_flags,
           retrieval_flags.
    Writes: proposer_decision, proposer_flags.

    Follows the same flag-don't-fail pattern as Nodes 1 and 2: always returns
    a state update, never raises.
    """
    req = state["extracted_request"]
    chunks = state.get("retrieved_policy_chunks", [])
    extraction_flags = state.get("extraction_flags", [])
    retrieval_flags = state.get("retrieval_flags", [])

    flags_section = format_flags_section(extraction_flags, retrieval_flags)
    request_section = format_request_section(req)
    policy_section = format_policy_section(chunks)

    structured_llm = get_llm().with_structured_output(ProposerDecision)
    messages = PROPOSER_PROMPT.format_messages(
        flags_section=flags_section,
        request_section=request_section,
        policy_section=policy_section,
    )

    try:
        decision = structured_llm.invoke(messages)
    except Exception:
        retry_messages = messages + [SystemMessage(content=PROPOSER_RETRY_HINT)]
        try:
            decision = structured_llm.invoke(retry_messages)
        except Exception:
            # Both attempts failed -- return a safe placeholder flagged for
            # human review rather than crashing the graph.
            decision = ProposerDecision(
                proposed_decision="need_more_info",
                rationale="Proposer node failed to produce a structured output. Human review required.",
                criteria_met=[],
                criteria_unmet=[],
                policy_citations=[],
                confidence=0.0,
            )
            return {
                "proposer_decision": decision,
                "proposer_flags": ["proposer_error"],
            }

    return {
        "proposer_decision": decision,
        "proposer_flags": _check_flags(decision, retrieval_flags),
    }


def _check_flags(decision: ProposerDecision, retrieval_flags: list[str]) -> list[str]:
    flags = []
    if decision.confidence < 0.6:
        flags.append("low_confidence_proposal")
    # If we had no relevant policy and the proposer still approved, flag it --
    # that's a sign it may have reasoned from general knowledge, not policy text.
    if "no_relevant_policy_found" in retrieval_flags and decision.proposed_decision == "approve":
        flags.append("approval_without_relevant_policy")
    return flags
