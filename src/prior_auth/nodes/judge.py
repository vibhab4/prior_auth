from langchain_core.messages import SystemMessage

from prior_auth.llm import get_llm
from prior_auth.prompts.critique import format_proposal_section
from prior_auth.prompts.judge import (
    JUDGE_PROMPT,
    JUDGE_RETRY_HINT,
    format_critic_section,
    format_flags_for_judge,
)
from prior_auth.prompts.propose import format_policy_section, format_request_section
from prior_auth.schemas.decision import FinalDecision
from prior_auth.state import PriorAuthState

# Below this confidence, always require human review regardless of other signals.
HUMAN_REVIEW_CONFIDENCE_THRESHOLD = 0.6


def judge_decision(state: PriorAuthState) -> dict:
    """Synthesize proposer + critic outputs into the final coverage determination.

    The judge is the only node that sets requires_human_review. It centralizes
    all routing logic: upstream flags, proposer/critic agreement, and its own
    confidence all feed into the human_review decision.

    Reads: extracted_request, retrieved_policy_chunks, proposer_decision,
           critic_feedback, + all flag fields.
    Writes: final_decision, judge_flags.

    Follows the same flag-don't-fail pattern as all prior nodes.
    """
    req = state["extracted_request"]
    chunks = state.get("retrieved_policy_chunks", [])
    proposal = state["proposer_decision"]
    feedback = state["critic_feedback"]

    extraction_flags = state.get("extraction_flags", [])
    retrieval_flags = state.get("retrieval_flags", [])
    proposer_flags = state.get("proposer_flags", [])
    critic_flags = state.get("critic_flags", [])

    flags_section = format_flags_for_judge(
        extraction_flags, retrieval_flags, proposer_flags, critic_flags
    )
    request_section = format_request_section(req)
    policy_section = format_policy_section(chunks)
    proposal_section = format_proposal_section(proposal)
    critic_section = format_critic_section(feedback)

    structured_llm = get_llm().with_structured_output(FinalDecision)
    messages = JUDGE_PROMPT.format_messages(
        flags_section=flags_section,
        request_section=request_section,
        policy_section=policy_section,
        proposal_section=proposal_section,
        critic_section=critic_section,
    )

    try:
        decision = structured_llm.invoke(messages)
    except Exception:
        retry_messages = messages + [SystemMessage(content=JUDGE_RETRY_HINT)]
        try:
            decision = structured_llm.invoke(retry_messages)
        except Exception:
            decision = FinalDecision(
                final_decision="need_more_info",
                final_rationale="Judge node failed to produce a structured output. Human review required.",
                key_factors=[],
                confidence=0.0,
                requires_human_review=True,
                human_review_reasons=["judge_error: structured output failed after retry"],
            )
            return {
                "final_decision": decision,
                "judge_flags": ["judge_error"],
            }

    # Apply deterministic guard: if any hard-trigger conditions are present,
    # force requires_human_review=True even if the LLM didn't set it.
    all_flags = extraction_flags + retrieval_flags + proposer_flags + critic_flags
    hard_trigger_reasons = _get_hard_trigger_reasons(
        all_flags, feedback, proposal, decision
    )
    if hard_trigger_reasons and not decision.requires_human_review:
        decision = decision.model_copy(
            update={
                "requires_human_review": True,
                "human_review_reasons": hard_trigger_reasons,
            }
        )

    return {
        "final_decision": decision,
        "judge_flags": _check_flags(decision),
    }


def _get_hard_trigger_reasons(
    all_flags: list[str],
    feedback,
    proposal,
    decision: FinalDecision,
) -> list[str]:
    """Return reasons that force human review, regardless of LLM output."""
    reasons = []
    error_flags = [f for f in all_flags if f.endswith("_error")]
    if error_flags:
        reasons.append(f"Upstream errors detected: {', '.join(error_flags)}")
    if "no_relevant_policy_found" in all_flags:
        reasons.append("No relevant coverage policy was retrieved for this request")
    if "critic_suggests_different_decision" in all_flags:
        reasons.append(
            f"Proposer ({proposal.proposed_decision}) and critic "
            f"({feedback.suggested_decision}) suggested different decisions"
        )
    if decision.confidence < HUMAN_REVIEW_CONFIDENCE_THRESHOLD:
        reasons.append(
            f"Final confidence ({decision.confidence:.2f}) is below the "
            f"human-review threshold ({HUMAN_REVIEW_CONFIDENCE_THRESHOLD})"
        )
    return reasons


def _check_flags(decision: FinalDecision) -> list[str]:
    flags = []
    if decision.requires_human_review:
        flags.append("requires_human_review")
    if decision.confidence < HUMAN_REVIEW_CONFIDENCE_THRESHOLD:
        flags.append("low_confidence_final")
    return flags
