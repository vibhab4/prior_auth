"""Integration tests for the extract_request node.

These call the real Claude API via with_structured_output, so they need
ANTHROPIC_API_KEY set (via .env). They're a basic, unit-level companion to
the trajectory-level eval set planned for Phase 4.
"""

from pathlib import Path

from prior_auth.nodes.extract import extract_request
from prior_auth.schemas.extraction import ExtractedRequest

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_requests"


def _run(case_file: str) -> dict:
    raw_text = (DATA_DIR / case_file).read_text()
    state = {"case_id": case_file, "raw_request_text": raw_text}
    return extract_request(state)


def test_clean_case_extracts_codes_with_no_flags():
    result = _run("case_001_clean.txt")
    extracted = result["extracted_request"]

    assert isinstance(extracted, ExtractedRequest)
    assert extracted.procedure_code is not None and "72148" in extracted.procedure_code
    assert extracted.diagnosis_code is not None and "M54.16" in extracted.diagnosis_code
    assert result["extraction_flags"] == []


def test_missing_diagnosis_code_is_flagged():
    result = _run("case_002_missing_dx_code.txt")
    extracted = result["extracted_request"]

    assert extracted.procedure_code is not None and "29881" in extracted.procedure_code
    assert extracted.diagnosis_code is None
    assert "missing_diagnosis_code" in result["extraction_flags"]


def test_sparse_case_is_low_confidence():
    result = _run("case_003_sparse.txt")
    extracted = result["extracted_request"]

    assert extracted.extraction_confidence < 0.6
    assert "low_confidence_extraction" in result["extraction_flags"]
