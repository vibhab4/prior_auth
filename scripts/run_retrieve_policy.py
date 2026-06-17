"""Run the two-node pipeline (extract -> retrieve) against a sample case.

Tests retrieve_policy in context -- builds a real ExtractedRequest via
extract_request, then runs retrieve_policy on the result. This matches the
order nodes will fire in the compiled graph.

Usage:
    python scripts/run_retrieve_policy.py [case_filename]

Defaults to case_001_clean.txt.
Requires: ANTHROPIC_API_KEY in .env, and data/vectorstore/ built by
          scripts/build_index.py.
"""

import sys
from pathlib import Path

from prior_auth.nodes.extract import extract_request
from prior_auth.nodes.retrieve_policy import retrieve_policy

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_requests"


def main() -> None:
    case_file = sys.argv[1] if len(sys.argv) > 1 else "case_001_clean.txt"
    case_path = DATA_DIR / case_file
    case_id = case_path.stem

    state: dict = {
        "case_id": case_id,
        "raw_request_text": case_path.read_text(),
    }

    # Node 1: extract
    print(f"--- {case_id}: extract_request ---")
    extract_result = extract_request(state)
    state.update(extract_result)
    print(state["extracted_request"].model_dump_json(indent=2))
    print("extraction_flags:", extract_result["extraction_flags"])

    # Node 2: retrieve
    print(f"\n--- {case_id}: retrieve_policy ---")
    retrieve_result = retrieve_policy(state)
    state.update(retrieve_result)

    chunks = retrieve_result["retrieved_policy_chunks"]
    print(f"retrieval_flags: {retrieve_result['retrieval_flags']}")
    print(f"retrieved {len(chunks)} chunk(s):\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] score={chunk.relevance_score:.3f} | {chunk.source_document}")
        print(f"       section: {chunk.section}")
        print(f"       subsection: {chunk.subsection}")
        print(f"       text[:200]: {chunk.chunk_text[:200].strip()!r}")
        print()


if __name__ == "__main__":
    main()
