"""Tests for the propose_coverage node (Node 3).

These tests make real Anthropic API calls -- they require ANTHROPIC_API_KEY
in the environment. Run them with:

    pytest tests/test_propose.py -v

They also require the vectorstore to be built (scripts/build_index.py).

Strategy: assert structural correctness (required fields present, valid enum
values, citation count > 0) rather than asserting specific decisions, since
LLM outputs vary between runs. The key invariants are:
  - The node never crashes, even on flagged/sparse cases
  - The output always conforms to ProposerDecision schema
  - Citations are present for on-target cases (the model cited something)
  - The flagging logic (proposer_flags) fires correctly
"""

import pytest

from prior_auth.nodes.extract import extract_request
from prior_auth.nodes.propose import propose_coverage
from prior_auth.nodes.retrieve_policy import retrieve_policy
from prior_auth.schemas.decision import ProposerDecision
from prior_auth.schemas.extraction import ExtractedRequest
from prior_auth.schemas.policy import PolicyChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_full_pipeline(raw_text: str, case_id: str = "test") -> dict:
    """Run all three nodes in sequence and return the final state."""
    state: dict = {"case_id": case_id, "raw_request_text": raw_text}
    state.update(extract_request(state))
    state.update(retrieve_policy(state))
    state.update(propose_coverage(state))
    return state


def _make_state_with_extraction(extracted: ExtractedRequest, chunks: list[PolicyChunk]) -> dict:
    """Build a minimal state dict for calling propose_coverage directly."""
    return {
        "case_id": "test",
        "raw_request_text": "",
        "extracted_request": extracted,
        "extraction_flags": [],
        "retrieved_policy_chunks": chunks,
        "retrieval_flags": [],
    }


# ---------------------------------------------------------------------------
# Case text fixtures
# ---------------------------------------------------------------------------

CASE_001_TEXT = """\
PRIOR AUTHORIZATION REQUEST

Patient: [SYNTHETIC] Jordan T., DOB 03/14/1978 (Age 48), Sex: M
Requesting Provider: Dr. A. Rivera, Orthopedic Associates
Requested Service: Lumbar MRI without contrast, outpatient
Requested CPT: 72148 (MRI lumbar spine without contrast)
Diagnosis: Lumbar radiculopathy, suspected L5-S1 disc herniation
(ICD-10: M54.16 - radiculopathy, lumbar region)

Clinical Notes:
Patient presents with 14 weeks of progressive low back pain radiating
to the left posterior leg, consistent with L5 distribution. Pain is
8/10, worsens with sitting and forward flexion. Patient completed
6 weeks of physical therapy (documented) and a trial of NSAIDs with
minimal improvement. Straight leg raise positive on left at 30 degrees.
No red flag symptoms. Conservative management has failed.
"""

CASE_003_TEXT = """\
REFERRAL NOTE

Patient: [SYNTHETIC] Alex P., Age 34, Sex: F
Provider: Dr. B. Chen

Requesting imaging study for shoulder pain.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_proposer_output_conforms_to_schema():
    """The proposer must always return a valid ProposerDecision object."""
    state = _run_full_pipeline(CASE_001_TEXT, "case_001")
    decision = state.get("proposer_decision")

    assert decision is not None
    assert isinstance(decision, ProposerDecision)
    assert decision.proposed_decision in ("approve", "deny", "need_more_info")
    assert isinstance(decision.rationale, str) and len(decision.rationale) > 50
    assert isinstance(decision.criteria_met, list)
    assert isinstance(decision.criteria_unmet, list)
    assert isinstance(decision.policy_citations, list)
    assert 0.0 <= decision.confidence <= 1.0


def test_strong_case_has_citations():
    """A well-documented case (case_001) should produce at least one policy citation."""
    state = _run_full_pipeline(CASE_001_TEXT, "case_001")
    decision = state["proposer_decision"]

    assert len(decision.policy_citations) > 0, (
        "Expected at least one policy citation for a well-documented lumbar MRI case"
    )


def test_strong_case_not_denied():
    """case_001 is a textbook approvable request -- it should not come back as deny."""
    state = _run_full_pipeline(CASE_001_TEXT, "case_001")
    decision = state["proposer_decision"]

    assert decision.proposed_decision != "deny", (
        f"case_001 (strong lumbar MRI case) should not be denied. "
        f"Got: {decision.proposed_decision}. Rationale: {decision.rationale[:300]}"
    )


def test_sparse_case_does_not_crash():
    """case_003 (vague shoulder, no codes) has no_relevant_policy_found -- node must not crash."""
    state = _run_full_pipeline(CASE_003_TEXT, "case_003")
    decision = state.get("proposer_decision")

    assert decision is not None
    assert decision.proposed_decision in ("approve", "deny", "need_more_info")
    # A vague case with no relevant policy should NOT be approved
    assert decision.proposed_decision != "approve" or (
        "approval_without_relevant_policy" in state.get("proposer_flags", [])
    ), "If approved without relevant policy, that flag must be set"


def test_proposer_flags_set_for_no_relevant_policy_approval():
    """If proposer approves despite no_relevant_policy_found, flag it."""
    # Craft a state where retrieval_flags contains no_relevant_policy_found
    # and force an ExtractedRequest so we can call propose_coverage directly.
    extracted = ExtractedRequest(
        procedure_description="Shoulder MRI",
        procedure_code=None,
        diagnosis_description="Shoulder pain",
        diagnosis_code=None,
        clinical_justification="Vague shoulder pain, no specifics.",
        extraction_confidence=0.3,
    )
    # Use a minimal PolicyChunk with low relevance to simulate the no-match scenario
    weak_chunk = PolicyChunk(
        source_document="mri_general_NCD_220_2_UHC.txt",
        source_title="UHC NCD 220.2 General MRI",
        section="COVERAGE",
        subsection=None,
        chunk_text="MRI is covered for medically necessary indications.",
        relevance_score=0.55,
    )
    state = {
        "case_id": "test_flag",
        "raw_request_text": "",
        "extracted_request": extracted,
        "extraction_flags": ["missing_procedure_code", "missing_diagnosis_code", "low_confidence_extraction"],
        "retrieved_policy_chunks": [weak_chunk],
        "retrieval_flags": ["no_relevant_policy_found"],
    }
    result = propose_coverage(state)
    decision = result["proposer_decision"]
    flags = result["proposer_flags"]

    assert decision is not None
    # If approved, the guard flag must be set
    if decision.proposed_decision == "approve":
        assert "approval_without_relevant_policy" in flags


def test_proposer_flags_returned_as_list():
    """proposer_flags must always be a list (never None or missing)."""
    state = _run_full_pipeline(CASE_001_TEXT, "case_001")
    flags = state.get("proposer_flags")
    assert isinstance(flags, list)
