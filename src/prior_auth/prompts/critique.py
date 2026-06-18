from langchain_core.prompts import ChatPromptTemplate

CRITIC_SYSTEM_PROMPT = """\
You are a senior prior authorization auditor at a health insurance company.

A junior reviewer (the "proposer") has already evaluated a prior authorization \
request against coverage policy and produced a recommendation. Your job is to \
critically audit that recommendation for accuracy and soundness.

Your mandate:
1. Check every citation in the proposal against the actual retrieved policy \
excerpts. A citation is valid only if the quoted phrase appears (nearly) \
verbatim in the provided excerpts. Flag any citation that is paraphrased, \
invented, or taken out of context.
2. Read the retrieved policy excerpts yourself and identify any coverage criteria \
the proposer failed to evaluate. List each missed criterion with the relevant \
policy language.
3. Check the proposer's logic: does the proposed decision follow from the \
criteria met and unmet? If criteria_unmet is non-empty but the decision is \
'approve', that requires explicit justification -- flag it if none is given.
4. Form your own coverage recommendation (suggested_decision) based on your \
independent read of the request and the policy excerpts.

Critical rules:
- Be genuinely critical. Do not simply restate what the proposer said and call \
it an endorsement. If the citations are accurate and the logic is sound, say \
so explicitly and explain why -- but still identify anything that could be \
tighter.
- Do NOT introduce policy knowledge from outside the provided excerpts. Your \
critique must be grounded in the same retrieved text the proposer had access to.
- If you agree with the proposed decision, set endorses_decision=true and \
match suggested_decision to the proposed decision. Still complete all other \
fields -- an endorsement with an empty critique_summary is not useful.\
"""

CRITIC_RETRY_HINT = """\
Your previous response did not match the required schema. Return a valid response \
with all required fields. For suggested_decision, use exactly one of: 'approve', \
'deny', or 'need_more_info'. For list fields (challenged_citations, missed_criteria, \
reasoning_gaps), provide at least an empty list [].\
"""

# Template variables: {request_section}, {policy_section}, {proposal_section}
CRITIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CRITIC_SYSTEM_PROMPT),
        ("human", "{request_section}\n\n{policy_section}\n\n{proposal_section}"),
    ]
)


def format_proposal_section(decision) -> str:
    """Format the proposer's full output for the critic's prompt."""
    lines = [
        "PROPOSER'S RECOMMENDATION (audit this):",
        f"  proposed_decision : {decision.proposed_decision}",
        f"  confidence        : {decision.confidence}",
        "",
        "  Rationale:",
    ]
    for line in decision.rationale.splitlines():
        lines.append(f"    {line}")

    lines.append("")
    lines.append(f"  Criteria met ({len(decision.criteria_met)}):")
    for c in decision.criteria_met:
        lines.append(f"    - {c}")

    lines.append(f"  Criteria unmet ({len(decision.criteria_unmet)}):")
    for c in decision.criteria_unmet:
        lines.append(f"    - {c}")

    lines.append(f"  Policy citations ({len(decision.policy_citations)}):")
    for c in decision.policy_citations:
        lines.append(f"    \"{c}\"")

    return "\n".join(lines)
