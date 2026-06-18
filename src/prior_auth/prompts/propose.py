from langchain_core.prompts import ChatPromptTemplate

PROPOSER_SYSTEM_PROMPT = """\
You are a prior authorization coverage reviewer at a health insurance company.

Your task is to evaluate a prior authorization request against the provided \
policy excerpts and produce a structured coverage recommendation.

Critical rules:
1. Ground your reasoning ONLY in the policy excerpts provided below the \
request. Do not use your general medical or insurance knowledge to fill in \
coverage criteria that aren't stated in the excerpts.
2. For each relevant coverage criterion you find in the policy text, \
explicitly state whether the request meets it, doesn't meet it, or the \
information needed to evaluate it is absent from the request.
3. If the retrieved policy excerpts are flagged as low-relevance or are not \
directly applicable to this procedure/diagnosis, lean toward 'need_more_info' \
rather than approving or denying on policy that doesn't cleanly fit.
4. Use 'need_more_info' when a specific required criterion (e.g. "6 weeks of \
conservative therapy") cannot be evaluated from the available documentation -- \
not as a hedge when the overall case is strong.
5. For policy_citations, quote exact phrases from the provided excerpts. \
Do not paraphrase.
6. Set confidence lower when: policy excerpts don't directly address this \
procedure, key clinical details are missing, or the case is genuinely \
borderline.\
"""

PROPOSER_RETRY_HINT = """\
Your previous response did not match the required schema. Return a valid \
response with all required fields. For proposed_decision, use exactly one of: \
'approve', 'deny', or 'need_more_info'. For list fields (criteria_met, \
criteria_unmet, policy_citations), provide at least an empty list [].\
"""

# Template variables: {flags_section}, {request_section}, {policy_section}
PROPOSER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", PROPOSER_SYSTEM_PROMPT),
        ("human", "{flags_section}\n\n{request_section}\n\n{policy_section}"),
    ]
)


def format_flags_section(extraction_flags: list[str], retrieval_flags: list[str]) -> str:
    """Format active flags so the proposer is aware of upstream data quality issues."""
    all_flags = extraction_flags + retrieval_flags
    if not all_flags:
        return "UPSTREAM FLAGS: None"
    lines = ["UPSTREAM FLAGS (data quality issues from prior nodes):"]
    for flag in all_flags:
        lines.append(f"  - {flag}")
    return "\n".join(lines)


def format_request_section(req) -> str:
    """Format the extracted request fields for the prompt."""
    lines = [
        "PRIOR AUTHORIZATION REQUEST:",
        f"  Procedure : {req.procedure_description}"
        + (f" (CPT {req.procedure_code})" if req.procedure_code else " (no CPT code)"),
        f"  Diagnosis : {req.diagnosis_description}"
        + (f" (ICD-10 {req.diagnosis_code})" if req.diagnosis_code else " (no ICD-10 code)"),
    ]
    if req.patient_age or req.patient_sex:
        age = str(req.patient_age) if req.patient_age else "unknown"
        sex = req.patient_sex if req.patient_sex else "unknown"
        lines.append(f"  Patient   : age {age}, sex {sex}")
    lines.append(f"  Clinical justification:\n    {req.clinical_justification}")
    return "\n".join(lines)


def format_policy_section(chunks: list) -> str:
    """Format retrieved policy chunks as numbered excerpts with citation metadata."""
    if not chunks:
        return "POLICY EXCERPTS: None retrieved."

    lines = ["POLICY EXCERPTS (evaluate the request against these ONLY):"]
    for i, chunk in enumerate(chunks, 1):
        header_parts = [f"Source: {chunk.source_title} ({chunk.source_document})"]
        if chunk.section:
            header_parts.append(f"Section: {chunk.section}")
        if chunk.subsection:
            header_parts.append(f"Subsection: {chunk.subsection}")
        if chunk.relevance_score is not None:
            header_parts.append(f"Relevance score: {chunk.relevance_score:.2f}")

        lines.append(f"\n[Excerpt {i}]")
        for part in header_parts:
            lines.append(f"  {part}")
        lines.append("  ---")
        # Indent the chunk text so it's visually distinct
        for line in chunk.chunk_text.splitlines():
            lines.append(f"  {line}")

    return "\n".join(lines)
