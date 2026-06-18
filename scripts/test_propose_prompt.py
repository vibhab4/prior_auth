"""One-off manual test: run the proposer prompt against case_001 and print
the raw LLM output. Used to inspect what Claude returns before locking in
the schema and writing the node.

Usage:
    python scripts/test_propose_prompt.py

Requires: ANTHROPIC_API_KEY in .env, data/vectorstore/ built.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from prior_auth.graph import build_graph
from prior_auth.llm import get_llm
from prior_auth.prompts.propose import (
    PROPOSER_PROMPT,
    format_flags_section,
    format_policy_section,
    format_request_section,
)
from prior_auth.schemas.decision import ProposerDecision

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_requests"


def main():
    # Run Nodes 1+2 to get extracted request + policy chunks
    app = build_graph()
    case_path = DATA_DIR / "case_001_clean.txt"
    state = app.invoke(
        {"case_id": "case_001", "raw_request_text": case_path.read_text()}
    )

    req = state["extracted_request"]
    chunks = state["retrieved_policy_chunks"]
    extraction_flags = state.get("extraction_flags", [])
    retrieval_flags = state.get("retrieval_flags", [])

    print("=== UPSTREAM STATE ===")
    print(f"Procedure : {req.procedure_description} ({req.procedure_code})")
    print(f"Diagnosis : {req.diagnosis_description} ({req.diagnosis_code})")
    print(f"Extraction flags: {extraction_flags}")
    print(f"Retrieval flags : {retrieval_flags}")
    print(f"Chunks retrieved: {len(chunks)}")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] score={c.relevance_score:.3f} | {c.source_document} | {c.subsection or c.section}")

    # Build the prompt
    flags_section = format_flags_section(extraction_flags, retrieval_flags)
    request_section = format_request_section(req)
    policy_section = format_policy_section(chunks)

    print("\n=== FULL PROMPT (human turn) ===")
    print(flags_section)
    print()
    print(request_section)
    print()
    print(policy_section[:2000], "...(truncated)" if len(policy_section) > 2000 else "")

    # Call LLM with structured output
    print("\n=== PROPOSER OUTPUT ===")
    structured_llm = get_llm().with_structured_output(ProposerDecision)
    messages = PROPOSER_PROMPT.format_messages(
        flags_section=flags_section,
        request_section=request_section,
        policy_section=policy_section,
    )
    result = structured_llm.invoke(messages)

    print(f"proposed_decision : {result.proposed_decision}")
    print(f"confidence        : {result.confidence}")
    print(f"\nrationale:\n{result.rationale}")
    print(f"\ncriteria_met ({len(result.criteria_met)}):")
    for c in result.criteria_met:
        print(f"  - {c}")
    print(f"\ncriteria_unmet ({len(result.criteria_unmet)}):")
    for c in result.criteria_unmet:
        print(f"  - {c}")
    print(f"\npolicy_citations ({len(result.policy_citations)}):")
    for c in result.policy_citations:
        print(f"  \"{c}\"")


if __name__ == "__main__":
    main()
