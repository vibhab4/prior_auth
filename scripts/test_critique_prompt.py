"""One-off manual test: run the critic prompt against case_001 and print output.
Used to inspect what Claude returns before writing the critique_proposal node.

Usage:
    python scripts/test_critique_prompt.py

Requires: ANTHROPIC_API_KEY in .env, data/vectorstore/ built.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from prior_auth.graph import build_graph
from prior_auth.llm import get_llm
from prior_auth.prompts.critique import CRITIC_PROMPT, format_proposal_section
from prior_auth.prompts.propose import format_policy_section, format_request_section
from prior_auth.schemas.decision import CriticFeedback

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_requests"


def main():
    # Run Nodes 1+2+3 to get the full upstream state
    app = build_graph()
    case_path = DATA_DIR / "case_001_clean.txt"
    state = app.invoke(
        {"case_id": "case_001", "raw_request_text": case_path.read_text()}
    )

    req = state["extracted_request"]
    chunks = state["retrieved_policy_chunks"]
    decision = state["proposer_decision"]

    print("=== PROPOSER OUTPUT (input to critic) ===")
    print(f"proposed_decision : {decision.proposed_decision}")
    print(f"confidence        : {decision.confidence}")
    print(f"criteria_met      : {len(decision.criteria_met)}")
    print(f"criteria_unmet    : {len(decision.criteria_unmet)}")
    print(f"citations         : {len(decision.policy_citations)}")

    request_section = format_request_section(req)
    policy_section = format_policy_section(chunks)
    proposal_section = format_proposal_section(decision)

    print("\n=== CRITIC OUTPUT ===")
    structured_llm = get_llm().with_structured_output(CriticFeedback)
    messages = CRITIC_PROMPT.format_messages(
        request_section=request_section,
        policy_section=policy_section,
        proposal_section=proposal_section,
    )
    result = structured_llm.invoke(messages)

    print(f"endorses_decision   : {result.endorses_decision}")
    print(f"suggested_decision  : {result.suggested_decision}")
    print(f"confidence          : {result.confidence}")
    print(f"\ncritique_summary:\n{result.critique_summary}")
    print(f"\nchallenged_citations ({len(result.challenged_citations)}):")
    for c in result.challenged_citations:
        print(f"  - {c}")
    print(f"\nmissed_criteria ({len(result.missed_criteria)}):")
    for c in result.missed_criteria:
        print(f"  - {c}")
    print(f"\nreasoning_gaps ({len(result.reasoning_gaps)}):")
    for g in result.reasoning_gaps:
        print(f"  - {g}")


if __name__ == "__main__":
    main()
