from typing import Literal, Optional

from pydantic import BaseModel, Field


class ProposerDecision(BaseModel):
    """Initial coverage recommendation produced by the proposer node.

    The proposer is the first LLM reasoning step: it evaluates the extracted
    request against retrieved policy excerpts and outputs a structured proposal.
    This is NOT the final decision -- the critic (Node 4) will challenge it and
    the judge (Node 5) will make the final call.
    """

    proposed_decision: Literal["approve", "deny", "need_more_info"] = Field(
        description=(
            "The proposed coverage determination. Use 'need_more_info' only when "
            "a specific required criterion cannot be evaluated from the available "
            "information -- not as a general hedge when the case has flags."
        )
    )

    rationale: str = Field(
        description=(
            "Step-by-step reasoning that walks through each relevant policy "
            "criterion and whether the request meets it. Reference policy text "
            "inline. This is the core of the auditable explanation."
        )
    )

    criteria_met: list[str] = Field(
        description=(
            "List of specific coverage criteria the request satisfies, each "
            "described in one sentence. Empty list if none are met."
        )
    )

    criteria_unmet: list[str] = Field(
        description=(
            "List of specific coverage criteria the request does NOT meet or "
            "cannot be evaluated from the available information. Include why "
            "each criterion is unmet or unclear."
        )
    )

    policy_citations: list[str] = Field(
        description=(
            "Short direct quotes from the retrieved policy excerpts that are "
            "most relevant to the decision. Each citation should be a verbatim "
            "phrase or sentence from the policy text, not a paraphrase."
        )
    )

    confidence: float = Field(
        description=(
            "Self-assessed confidence (0.0-1.0) in this proposal. Use a lower "
            "value when the policy excerpts don't cleanly address this request, "
            "when key clinical information is absent, or when the case is borderline."
        )
    )


class CriticFeedback(BaseModel):
    """Adversarial review of the proposer's coverage recommendation.

    The critic reads the proposer's full output alongside the original policy
    excerpts and tries to find flaws: inaccurate citations, missed criteria,
    and logical gaps in the rationale. Produced by the critique_proposal node
    (Node 4). The judge (Node 5) weighs both the proposal and this feedback
    to make the final decision.
    """

    endorses_decision: bool = Field(
        description=(
            "True if the critic agrees that the proposed decision is correct "
            "and well-supported. False if the critic believes the decision is "
            "wrong, unsupported, or that a different decision is warranted."
        )
    )

    critique_summary: str = Field(
        description=(
            "One-paragraph overall assessment of the proposer's reasoning. "
            "Should explain specifically what the proposer got right and wrong, "
            "not just 'the proposal is good' or 'the proposal is flawed'."
        )
    )

    challenged_citations: list[str] = Field(
        description=(
            "Citations from the proposer that the critic believes are inaccurate, "
            "overstated, taken out of context, or not actually present in the "
            "retrieved policy excerpts. Each entry should quote the challenged "
            "citation and explain why it's problematic. Empty list if no citations "
            "are challenged."
        )
    )

    missed_criteria: list[str] = Field(
        description=(
            "Coverage criteria mentioned in the retrieved policy excerpts that the "
            "proposer did not evaluate. Each entry should name the criterion and "
            "quote the relevant policy language. Empty list if no criteria were missed."
        )
    )

    reasoning_gaps: list[str] = Field(
        description=(
            "Logical gaps, unsupported leaps, or inconsistencies in the proposer's "
            "rationale -- e.g., concluding 'approve' while listing unmet criteria, "
            "or asserting a criterion is met without citing policy support. "
            "Empty list if the reasoning is sound."
        )
    )

    suggested_decision: Literal["approve", "deny", "need_more_info"] = Field(
        description=(
            "The critic's own coverage recommendation after reviewing the proposal. "
            "If the critic agrees with the proposer, this should match proposed_decision. "
            "If the critic disagrees, state the decision the critic believes is correct."
        )
    )

    confidence: float = Field(
        description=(
            "Critic's confidence (0.0-1.0) in its own suggested_decision. Use a lower "
            "value when the critique is itself uncertain -- e.g., when the policy "
            "excerpts are ambiguous or the clinical picture is genuinely borderline."
        )
    )
