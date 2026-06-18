"""One-off manual test: run the judge prompt against case_001 and print output.
Used to inspect what Claude returns before writing the judge_decision node.

Usage:
    python scripts/test_judge_prompt.py

Requires: ANTHROPIC_API_KEY in .env, data/vectorstore/ built.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from prior_auth.graph import build_graph
from prior_auth.llm import get_llm
from prior_auth.prompts.critique import format_proposal_section
from prior_auth.prompts.judge import (
    JUDGE_PROMPT,
    format_critic_section,
    format_flags_for_judge,
)
from prior_auth.prompts.propose import format_policy_section, format_request_section
from prior_auth.schemas.decision import FinalDecision

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_requests"


def main():
    app = build_graph()
    case_path = DATA_DIR / "case_001_clean.txt"
    state = app.invoke(
        {"case_id": "case_001", "raw_request_text": case_path.read_text()}
    )

    req = state["extracted_request"]
    chunks = state["retrieved_policy_chunks"]
    proposal = state["proposer_decision"]
    feedback = state["critic_feedback"]

    print("=== UPSTREAM SUMMARY ===")
    print(f"Proposer: {proposal.proposed_decision} / {proposal.confidence}")
    print(f"Critic  : endorses={feedback.endorses_decision} / suggests={feedback.suggested_decision}")
    print(f"Proposer flags: {state.get('proposer_flags', [])}")
    print(f"Critic flags  : {state.get('critic_flags', [])}")

    flags_section = format_flags_for_judge(
        state.get("extraction_flags", []),
        state.get("retrieval_flags", []),
        state.get("proposer_flags", []),
        state.get("critic_flags", []),
    )
    request_section = format_request_section(req)
    policy_section = format_policy_section(chunks)
    proposal_section = format_proposal_section(proposal)
    critic_section = format_critic_section(feedback)

    print("\n=== JUDGE OUTPUT ===")
    structured_llm = get_llm().with_structured_output(FinalDecision)
    messages = JUDGE_PROMPT.format_messages(
        flags_section=flags_section,
        request_section=request_section,
        policy_section=policy_section,
        proposal_section=proposal_section,
        critic_section=critic_section,
    )
    result = structured_llm.invoke(messages)

    print(f"final_decision        : {result.final_decision}")
    print(f"confidence            : {result.confidence}")
    print(f"requires_human_review : {result.requires_human_review}")
    print(f"human_review_reasons  : {result.human_review_reasons}")
    print(f"\nkey_factors ({len(result.key_factors)}):")
    for f in result.key_factors:
        print(f"  - {f}")
    print(f"\nfinal_rationale:\n{result.final_rationale}")


if __name__ == "__main__":
    main()
