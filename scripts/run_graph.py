"""Run the compiled LangGraph graph end-to-end against all sample cases.

This is the first time we exercise .invoke() on a compiled graph rather
than calling nodes directly. It's the definitive end-to-end smoke test for
the Phase 1 pipeline as it exists so far (extract -> retrieve).

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


if __name__ == "__main__":
    main()
