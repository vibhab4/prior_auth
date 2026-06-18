"""Run the compiled LangGraph graph end-to-end against all sample cases.

Exercises .invoke() on the full compiled graph (extract -> retrieve ->
propose). Prints node outputs for each case for visual inspection.

Usage:
    python scripts/run_graph.py

Requires: ANTHROPIC_API_KEY in .env, data/vectorstore/ built by
          scripts/build_index.py.
"""

from pathlib import Path

from prior_auth.graph import build_graph

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_requests"
CASES = [
    "case_001_clean.txt",
    "case_002_missing_dx_code.txt",
    "case_003_sparse.txt",
]


def main() -> None:
    app = build_graph()

    for case_file in CASES:
        case_path = DATA_DIR / case_file
        case_id = case_path.stem
        print(f"\n{'='*60}")
        print(f"CASE: {case_id}")
        print("=" * 60)

        initial_state = {
            "case_id": case_id,
            "raw_request_text": case_path.read_text(),
        }

        # .invoke() runs the full graph and returns the final state dict.
        # This is the idiomatic LangGraph entry point for single-shot runs.
        final_state = app.invoke(initial_state)

        # Extraction results
        req = final_state.get("extracted_request")
        if req:
            print(f"\n[Node 1 - extract_request]")
            print(f"  procedure : {req.procedure_description} ({req.procedure_code})")
            print(f"  diagnosis : {req.diagnosis_description} ({req.diagnosis_code})")
            print(f"  confidence: {req.extraction_confidence}")
        print(f"  extraction_flags: {final_state.get('extraction_flags', [])}")

        # Retrieval results
        chunks = final_state.get("retrieved_policy_chunks", [])
        print(f"\n[Node 2 - retrieve_policy]")
        print(f"  retrieval_flags: {final_state.get('retrieval_flags', [])}")
        print(f"  {len(chunks)} chunk(s) retrieved:")
        for i, chunk in enumerate(chunks, 1):
            print(
                f"    [{i}] score={chunk.relevance_score:.3f}"
                f" | {chunk.source_document}"
                f" | {chunk.subsection or chunk.section or 'N/A'}"
            )

        # Proposer results
        decision = final_state.get("proposer_decision")
        print(f"\n[Node 3 - propose_coverage]")
        print(f"  proposer_flags: {final_state.get('proposer_flags', [])}")
        if decision:
            print(f"  proposed_decision : {decision.proposed_decision}")
            print(f"  confidence        : {decision.confidence}")
            print(f"  criteria_met      : {len(decision.criteria_met)}")
            print(f"  criteria_unmet    : {len(decision.criteria_unmet)}")
            print(f"  policy_citations  : {len(decision.policy_citations)}")
            print(f"  rationale preview : {decision.rationale[:200]}...")

        # Critic results
        feedback = final_state.get("critic_feedback")
        print(f"\n[Node 4 - critique_proposal]")
        print(f"  critic_flags: {final_state.get('critic_flags', [])}")
        if feedback:
            print(f"  endorses_decision  : {feedback.endorses_decision}")
            print(f"  suggested_decision : {feedback.suggested_decision}")
            print(f"  confidence         : {feedback.confidence}")
            print(f"  challenged_cites   : {len(feedback.challenged_citations)}")
            print(f"  missed_criteria    : {len(feedback.missed_criteria)}")
            print(f"  reasoning_gaps     : {len(feedback.reasoning_gaps)}")
            print(f"  summary preview    : {feedback.critique_summary[:200]}...")

        # Final decision
        final = final_state.get("final_decision")
        print(f"\n[Node 5 - judge_decision]  *** FINAL OUTPUT ***")
        print(f"  judge_flags           : {final_state.get('judge_flags', [])}")
        if final:
            print(f"  final_decision        : {final.final_decision}")
            print(f"  confidence            : {final.confidence}")
            print(f"  requires_human_review : {final.requires_human_review}")
            if final.human_review_reasons:
                for r in final.human_review_reasons:
                    print(f"    reason: {r}")
            print(f"  key_factors ({len(final.key_factors)}):")
            for kf in final.key_factors:
                print(f"    - {kf}")
            print(f"  rationale preview     : {final.final_rationale[:300]}...")


if __name__ == "__main__":
    main()
