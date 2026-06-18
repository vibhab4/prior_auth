"""Tests for the judge_decision node (Node 5).

Real Anthropic API calls -- requires ANTHROPIC_API_KEY in the environment
and data/vectorstore/ built by scripts/build_index.py.

Run with: pytest tests/test_judge.py -v

Invariants tested:
  - judge_decision never crashes
  - Output always conforms to FinalDecision schema
  - Hard-trigger conditions force requires_human_review=True deterministically
  - case_001 (clean, strong case) produces approve without human_review
  - case_003 (no relevant policy) triggers requires_human_review=True
  - judge_flags reflect requires_human_review correctly
"""

import pytest

from prior_auth.nodes.extract import extract_request
from prior_auth.nodes.critique import critique_proposal
from prior_auth.nodes.judge import judge_decision
from prior_auth.nodes.propose import propose_coverage
from prior_auth.nodes.retrieve_policy import retrieve_policy
from prior_auth.schemas.decision import (
    CriticFeedback,
    FinalDecision,
    ProposerDecision,
)
from prior_auth.schemas.extraction import ExtractedRequest
from prior_auth.schemas.policy import PolicyChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_full_pipeline(raw_text: str, case_id: str = "test") -> dict:
    """Run all five nodes in sequence and return the final state."""
    state: dict = {"case_id": case_id, "raw_request_text": raw_text}
    state.update(extract_request(state))
    state.update(retrieve_policy(state))
    state.update(propose_coverage(state))
    state.update(critique_proposal(state))
    state.update(judge_decision(state))
    return state


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

def test_final_decision_conforms_to_schema():
    """judge_decision must always return a valid FinalDecision object."""
    state = _run_full_pipeline(CASE_001_TEXT, "case_001")
    final = state.get("final_decision")

    assert final is not None
    assert isinstance(final, FinalDecision)
    assert final.final_decision in ("approve", "deny", "need_more_info")
    assert isinstance(final.final_rationale, str) and len(final.final_rationale) > 100
    assert isinstance(final.key_factors, list) and len(final.key_factors) > 0
    assert 0.0 <= final.confidence <= 1.0
    assert isinstance(final.requires_human_review, bool)
    assert isinstance(final.human_review_reasons, list)


def test_clean_case_approved_no_human_review():
    """case_001 (strong, clean) should be approved without requiring human review."""
    state = _run_full_pipeline(CASE_001_TEXT, "case_001")
    final = state["final_decision"]

    assert final.final_decision == "approve", (
        f"Expected approve for strong case_001. Got: {final.final_decision}. "
        f"Rationale: {final.final_rationale[:300]}"
    )
    assert not final.requires_human_review, (
        f"case_001 should not require human review. Reasons: {final.human_review_reasons}"
    )


def test_no_relevant_policy_triggers_human_review():
    """case_003 (no relevant policy found) must trigger requires_human_review=True."""
    state = _run_full_pipeline(CASE_003_TEXT, "case_003")
    final = state["final_decision"]

    assert final.requires_human_review, (
        "case_003 has no_relevant_policy_found flag -- should require human review"
    )
    assert len(final.human_review_reasons) > 0


def test_hard_trigger_forces_human_review():
    """The deterministic guard must force requires_human_review=True for error flags,
    even if the LLM didn't set it."""
    extracted = ExtractedRequest(
        procedure_description="Lumbar MRI",
        procedure_code="72148",
        diagnosis_description="Lumbar radiculopathy",
        diagnosis_code="M54.16",
        clinical_justification="Failed PT, positive SLR.",
        extraction_confidence=0.95,
    )
    chunk = PolicyChunk(
        source_document="lumbar_mri_NIA_CG_044.txt",
        source_title="Lumbar Spine MRI Clinical Guidelines",
        section="INDICATIONS",
        subsection=None,
        chunk_text="Lumbar MRI is indicated after 6 weeks of failed conservative treatment.",
        relevance_score=0.77,
    )
    proposal = ProposerDecision(
        proposed_decision="approve",
        rationale="Criteria met.",
        criteria_met=["6 weeks PT documented"],
        criteria_unmet=[],
        policy_citations=["Lumbar MRI is indicated after 6 weeks of failed conservative treatment."],
        confidence=0.9,
    )
    feedback = CriticFeedback(
        endorses_decision=True,
        critique_summary="Proposal is well-supported.",
        challenged_citations=[],
        missed_criteria=[],
        reasoning_gaps=[],
        suggested_decision="approve",
        confidence=0.9,
    )
    state = {
        "case_id": "test_error_flag",
        "raw_request_text": "",
        "extracted_request": extracted,
        "extraction_flags": ["extraction_error"],  # hard-trigger flag
        "retrieved_policy_chunks": [chunk],
        "retrieval_flags": [],
        "proposer_decision": proposal,
        "proposer_flags": [],
        "critic_feedback": feedback,
        "critic_flags": [],
    }
    result = judge_decision(state)
    final = result["final_decision"]

    assert final.requires_human_review, (
        "extraction_error flag should force requires_human_review=True"
    )
    assert any("error" in r.lower() for r in final.human_review_reasons)


def test_judge_flags_reflect_human_review():
    """requires_human_review in judge_flags must match final_decision.requires_human_review."""
    state = _run_full_pipeline(CASE_001_TEXT, "case_001")
    final = state["final_decision"]
    flags = state.get("judge_flags", [])

    if final.requires_human_review:
        assert "requires_human_review" in flags
    else:
        assert "requires_human_review" not in flags


def test_final_rationale_is_self_contained():
    """final_rationale should be long enough to be a standalone explanation."""
    state = _run_full_pipeline(CASE_001_TEXT, "case_001")
    final = state["final_decision"]

    # A self-contained rationale should mention key facts from the case
    rationale_lower = final.final_rationale.lower()
    assert any(word in rationale_lower for word in ["policy", "criteria", "conservative", "mri"]), (
        "final_rationale should reference policy and clinical facts"
    )
    assert len(final.final_rationale) > 300, (
        f"final_rationale too short ({len(final.final_rationale)} chars) to be self-contained"
    )


def test_judge_does_not_crash_on_sparse_case():
    """judge_decision must not crash even for the vague case_003."""
    state = _run_full_pipeline(CASE_003_TEXT, "case_003")
    final = state.get("final_decision")

    assert final is not None
    assert final.final_decision in ("approve", "deny", "need_more_info")
