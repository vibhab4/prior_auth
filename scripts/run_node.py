"""Run a single LangGraph node against a sample case, without building or
compiling a graph. This is the pattern for developing/testing nodes one at
a time as the graph grows.

Usage:
    python scripts/run_node.py [case_filename]

Defaults to case_001_clean.txt. Case files live in data/synthetic_requests/.
"""

import sys
from pathlib import Path

from prior_auth.nodes.extract import extract_request

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_requests"


def main() -> None:
    case_file = sys.argv[1] if len(sys.argv) > 1 else "case_001_clean.txt"
    case_path = DATA_DIR / case_file
    case_id = case_path.stem

    state = {
        "case_id": case_id,
        "raw_request_text": case_path.read_text(),
    }

    result = extract_request(state)

    print(f"--- {case_id} ---")
    print(result["extracted_request"].model_dump_json(indent=2))
    print("flags:", result["extraction_flags"])


if __name__ == "__main__":
    main()
