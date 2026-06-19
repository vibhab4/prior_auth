"""Demonstrate Phase 2: SQLite checkpointing and multi-turn "need more info" flow.

Shows the two-run lifecycle of a prior auth case that initially gets
"need_more_info" and is later resubmitted with the missing documentation.

Run 1 (initial submission):
    case_002 → need_more_info, requires_human_review=True
    State is saved to data/checkpoints.db keyed by thread_id="case_002"

Run 2 (follow-up submission):
    case_002_followup → now has ICD-10 code + OA grading + PT records
    Graph runs fresh against the complete submission
    Final decision should change (likely approve or cleaner need_more_info)

The key point: in a real system, Run 1's state (extracted request,
retrieved chunks, proposer/critic reasoning) would be preserved in the
database and could be reloaded by a case manager. The follow-up could then
re-run only the nodes that need new inputs, rather than starting from scratch.
For this demo we run the full graph both times to keep it simple.

Usage:
    python scripts/resume_case.py

Requires: ANTHROPIC_API_KEY in .env, data/vectorstore/ built.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from prior_auth.checkpointing import get_checkpointer
from prior_auth.graph import build_graph

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_requests"


def print_result(label: str, state: dict) -> None:
    final = state.get("final_decision")
    print(f"\n{'='*60}")
    print(f"{label}")
    print("=" * 60)
    if final:
        print(f"  final_decision        : {final.final_decision}")
        print(f"  confidence            : {final.confidence}")
        print(f"  requires_human_review : {final.requires_human_review}")
        if final.human_review_reasons:
            print(f"  human_review_reasons:")
            for r in final.human_review_reasons:
                print(f"    - {r}")
        print(f"  key_factors:")
        for kf in final.key_factors:
            print(f"    - {kf}")
        print(f"  rationale (first 400 chars):")
        print(f"    {final.final_rationale[:400]}...")


def main():
    print("Prior Auth Checkpointing Demo — case_002 multi-turn flow")
    print("State is persisted to data/checkpoints.db after each run.\n")

    with get_checkpointer() as checkpointer:
        app = build_graph(checkpointer=checkpointer)

        # --- Run 1: Initial submission (missing ICD code, OA grading) ---
        print("RUN 1: Initial submission (case_002_missing_dx_code)")
        print("  Expected: need_more_info — missing ICD code, OA grading absent")

        case_002_text = (DATA_DIR / "case_002_missing_dx_code.txt").read_text()
        config_run1 = {"configurable": {"thread_id": "case_002"}}

        state_run1 = app.invoke(
            {"case_id": "case_002", "raw_request_text": case_002_text},
            config=config_run1,
        )
        print_result("RUN 1 RESULT (initial submission)", state_run1)

        # Verify state was persisted
        saved = app.get_state(config_run1)
        print(f"\n[Checkpoint] State saved to SQLite.")
        print(f"  thread_id : case_002")
        print(f"  next nodes: {saved.next}")  # empty = graph completed
        print(f"  values keys: {list(saved.values.keys())}")

        # --- Run 2: Follow-up submission (complete documentation) ---
        print("\n\nRUN 2: Follow-up submission (case_002_followup)")
        print("  New info: ICD-10 M23.201, K-L Grade I OA, 8 weeks formal PT documented")
        print("  Expected: approve or cleaner need_more_info, lower human_review burden")

        case_002_followup_text = (DATA_DIR / "case_002_followup.txt").read_text()
        # New thread_id for the follow-up — keeps the two runs separately addressable
        config_run2 = {"configurable": {"thread_id": "case_002_followup"}}

        state_run2 = app.invoke(
            {"case_id": "case_002_followup", "raw_request_text": case_002_followup_text},
            config=config_run2,
        )
        print_result("RUN 2 RESULT (follow-up with complete documentation)", state_run2)

        # Show how to retrieve either run from the database later
        print("\n\n[Checkpoint] Both runs are now in the database.")
        print("  To load run 1 later: app.get_state({'configurable': {'thread_id': 'case_002'}})")
        print("  To load run 2 later: app.get_state({'configurable': {'thread_id': 'case_002_followup'}})")
        print("\n  This is how a case manager tool would reload a case:")
        reloaded = app.get_state(config_run1)
        reloaded_final = reloaded.values.get("final_decision")
        if reloaded_final:
            print(f"  Reloaded case_002 → final_decision: {reloaded_final.final_decision}")


if __name__ == "__main__":
    main()
