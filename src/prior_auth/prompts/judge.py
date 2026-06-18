from langchain_core.prompts import ChatPromptTemplate

JUDGE_SYSTEM_PROMPT = """\
You are the final decision-maker in a prior authorization review pipeline.

Two reviewers have already evaluated this case:
- The PROPOSER assessed the request against the retrieved coverage policy and \
produced an initial recommendation with rationale and citations.
- The CRITIC audited the proposer's recommendation, checking for citation \
accuracy, missed criteria, and logical gaps.

Your job is to synthesize both inputs and produce the definitive coverage \
determination.

Decision guidelines:
1. When the proposer and critic AGREE on a decision: lean toward that shared \
decision, adjusting only if you find a clear error both missed.
2. When they DISAGREE on the decision: be conservative. Lean toward \
'need_more_info' unless one position is clearly better supported by the \
retrieved policy excerpts. Explain specifically why you sided with one over \
the other.
3. If the critic identified valid challenges or missed criteria, factor them \
into your rationale -- don't simply restate the proposer's reasoning.
4. Set requires_human_review=True when any of the following apply:
   - Proposer and critic suggested different decisions
   - Your final confidence is below 0.6
   - The retrieval flags indicate no relevant policy was found
   - Any upstream error flag is present (extraction_error, retrieval_error, \
proposer_error, critic_error)

Critical rules (same as all prior nodes):
- Ground your final_rationale in the PROVIDED policy excerpts only. Do not \
use general medical or insurance knowledge to fill in coverage criteria not \
present in the retrieved text.
- final_rationale must be self-contained: a human reviewer reading only that \
field should understand the decision without needing to re-read the upstream \
node outputs.
- key_factors should be specific (e.g. "6 weeks of documented PT confirmed") \
not generic (e.g. "conservative treatment requirement").\
"""

JUDGE_RETRY_HINT = """\
Your previous response did not match the required schema. Return a valid \
response with all required fields. For final_decision, use exactly one of: \
'approve', 'deny', or 'need_more_info'. requires_human_review must be a \
boolean. For list fields (key_factors, human_review_reasons), provide at \
least an empty list [].\
"""

# Template variables: {flags_section}, {request_section}, {policy_section},
#                     {proposal_section}, {critic_section}
JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", JUDGE_SYSTEM_PROMPT),
        (
            "human",
            "{flags_section}\n\n{request_section}\n\n{policy_section}\n\n"
            "{proposal_section}\n\n{critic_section}",
        ),
    ]
)


def format_critic_section(feedback) -> str:
    """Format the critic's full output for the judge's prompt."""
    lines = [
        "CRITIC'S REVIEW (audit of the proposer's recommendation):",
        f"  endorses_decision  : {feedback.endorses_decision}",
        f"  suggested_decision : {feedback.suggested_decision}",
        f"  confidence         : {feedback.confidence}",
        "",
        "  Critique summary:",
    ]
    for line in feedback.critique_summary.splitlines():
        lines.append(f"    {line}")

    lines.append(f"\n  Challenged citations ({len(feedback.challenged_citations)}):")
    for c in feedback.challenged_citations:
        lines.append(f"    - {c}")

    lines.append(f"  Missed criteria ({len(feedback.missed_criteria)}):")
    for c in feedback.missed_criteria:
        lines.append(f"    - {c}")

    lines.append(f"  Reasoning gaps ({len(feedback.reasoning_gaps)}):")
    for g in feedback.reasoning_gaps:
        lines.append(f"    - {g}")

    return "\n".join(lines)


def format_flags_for_judge(
    extraction_flags: list[str],
    retrieval_flags: list[str],
    proposer_flags: list[str],
    critic_flags: list[str],
) -> str:
    """Consolidate all upstream flags into a single section for the judge."""
    all_flags = {
        "extraction": extraction_flags,
        "retrieval": retrieval_flags,
        "proposer": proposer_flags,
        "critic": critic_flags,
    }
    has_any = any(flags for flags in all_flags.values())
    if not has_any:
        return "UPSTREAM FLAGS: None — all nodes ran cleanly."

    lines = ["UPSTREAM FLAGS (weigh these when setting requires_human_review):"]
    for source, flags in all_flags.items():
        if flags:
            for flag in flags:
                lines.append(f"  [{source}] {flag}")
    return "\n".join(lines)
