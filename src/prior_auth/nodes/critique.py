from langchain_core.messages import SystemMessage

from prior_auth.llm import get_llm
from prior_auth.prompts.critique import (
    CRITIC_PROMPT,
    CRITIC_RETRY_HINT,
    format_proposal_section,
)
from prior_auth.prompts.propose import format_policy_section, format_request_section
from prior_auth.schemas.decision import CriticFeedback
from prior_auth.state import PriorAuthState


def critique_proposal(state: PriorAuthState) -> dict:
    """Adversarially review the proposer's coverage recommendation.

    Reads the proposer's full output (decision, rationale, citations, criteria)
    alongside the original policy excerpts and checks for: inaccurate citations,
    missed criteria, and logical gaps in the reasoning. Produces its own
    suggested_decision independently.

    Reads: extracted_request, retrieved_policy_chunks, proposer_decision.
    Writes: critic_feedback, critic_flags.

    Follows the same flag-don't-fail pattern as all prior nodes.
    """
    req = state["extracted_request"]
    chunks = state.get("retrieved_policy_chunks", [])
    decision = state["proposer_decision"]

    request_section = format_request_section(req)
    policy_section = format_policy_section(chunks)
    proposal_section = format_proposal_section(decision)

    structured_llm = get_llm().with_structured_output(CriticFeedback)
    messages = CRITIC_PROMPT.format_messages(
        request_section=request_section,
        policy_section=policy_section,
        proposal_section=proposal_section,
    )

    try:
        feedback = structured_llm.invoke(messages)
    except Exception:
        retry_messages = messages + [SystemMessage(content=CRITIC_RETRY_HINT)]
        try:
            feedback = structured_llm.invoke(retry_messages)
        except Exception:
            feedback = CriticFeedback(
                endorses_decision=False,
                critique_summary="Critic node failed to produce a structured output. Human review required.",
                challenged_citations=[],
                missed_criteria=[],
                reasoning_gaps=[],
                suggested_decision="need_more_info",
                confidence=0.0,
            )
            return {
                "critic_feedback": feedback,
                "critic_flags": ["critic_error"],
            }

    return {
        "critic_feedback": feedback,
        "critic_flags": _check_flags(feedback, decision),
    }


def _check_flags(feedback: CriticFeedback, decision) -> list[str]:
    flags = []
    if not feedback.endorses_decision:
        flags.append("critic_disagrees")
    # Disagreement on the actual decision (not just nitpicks) is more severe
    if feedback.suggested_decision != decision.proposed_decision:
        flags.append("critic_suggests_different_decision")
    if feedback.confidence < 0.6:
        flags.append("low_confidence_critique")
    return flags
