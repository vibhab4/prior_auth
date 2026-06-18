"""Tests for the critique_proposal node (Node 4).

Real Anthropic API calls -- requires ANTHROPIC_API_KEY in the environment
and data/vectorstore/ built by scripts/build_index.py.

Run with: pytest tests/test_critique.py -v

Same test philosophy as test_propose.py: assert structural invariants and
behavioral guarantees, not specific LLM outputs. Key invariants:
  - critique_proposal never crashes, even on flagged upstream state
  - Output always conforms to CriticFeedback schema
  - critic_flags are set correctly based on endorses_decision and
    suggested_decision agreement with proposed_decision
  - A strong proposal (case_001) should be endorsed
"""

import pytest

from prior_auth.nodes.critique import critique_proposal
from prior_auth.nodes.extract import extract_request
from prior_auth.nodes.propose import propose_coverage
from prior_auth.nodes.retrieve_policy import retrieve_policy
from prior_auth.schemas.decision import CriticFeedback, ProposerDecision
from prior_auth.schemas.extraction import ExtractedRequest
from prior_auth.schemas.policy import PolicyChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_full_pipeline(raw_text: str, case_id: str = "test") -> dict:
    """Run all four nodes in sequence and return the final state."""
    state: dict = {"case_id": case_id, "raw_request_text": raw_text}
    state.update(extract_request(state))
    state.update(retrieve_policy(state))
    state.update(propose_coverage(state))
    state.update(critique_proposal(state))
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

def test_critic_output_conforms_to_schema():
    """critique_proposal must always return a valid CriticFeedback object."""
    state = _run_full_pipeline(CASE_001_TEXT, "case_001")
    feedback = state.get("critic_feedback")

    assert feedback is not None
    assert isinstance(feedback, CriticFeedback)
    assert feedback.suggested_decision in ("approve", "deny", "need_more_info")
    assert isinstance(feedback.endorses_decision, bool)
    assert isinstance(feedback.critique_summary, str) and len(feedback.critique_summary) > 20
    assert isinstance(feedback.challenged_citations, list)
    assert isinstance(feedback.missed_criteria, list)
    assert isinstance(feedback.reasoning_gaps, list)
    assert 0.0 <= feedback.confidence <= 1.0


def test_strong_proposal_endorsed():
    """A well-supported proposal (case_001, approve) should be endorsed by the critic."""
    state = _run_full_pipeline(CASE_001_TEXT, "case_001")
    feedback = state["critic_feedback"]
    proposal = state["proposer_decision"]

    # For a clear-cut approvable case, the critic should agree
    assert feedback.endorses_decision, (
        f"Expected critic to endorse strong case_001 proposal. "
        f"Critique summary: {feedback.critique_summary[:300]}"
    )
    assert feedback.suggested_decision == proposal.proposed_decision


def test_critic_does_not_crash_on_sparse_case():
    """critique_proposal must not crash even when upstream flags are set (case_003)."""
    state = _run_full_pipeline(CASE_003_TEXT, "case_003")
    feedback = state.get("critic_feedback")

    assert feedback is not None
    assert feedback.suggested_decision in ("approve", "deny", "need_more_info")


def test_critic_disagrees_flag_set_correctly():
    """critic_disagrees flag must be set iff endorses_decision is False."""
    state = _run_full_pipeline(CASE_001_TEXT, "case_001")
    feedback = state["critic_feedback"]
    flags = state.get("critic_flags", [])

    if not feedback.endorses_decision:
        assert "critic_disagrees" in flags
    else:
        assert "critic_disagrees" not in flags


def test_different_decision_flag_set_correctly():
    """critic_suggests_different_decision flag must be set iff decisions diverge."""
    state = _run_full_pipeline(CASE_001_TEXT, "case_001")
    feedback = state["critic_feedback"]
    proposal = state["proposer_decision"]
    flags = state.get("critic_flags", [])

    if feedback.suggested_decision != proposal.proposed_decision:
        assert "critic_suggests_different_decision" in flags
    else:
        assert "critic_suggests_different_decision" not in flags


def test_critic_flags_returned_as_list():
    """critic_flags must always be a list, never None."""
    state = _run_full_pipeline(CASE_001_TEXT, "case_001")
    assert isinstance(state.get("critic_flags"), list)


def test_critic_with_fabricated_weak_proposal():
    """Critic should challenge a proposal that cites non-existent policy language."""
    # Build a state with a weak proposer output that has a fabricated citation
    extracted = ExtractedRequest(
        procedure_description="Lumbar MRI without contrast",
        procedure_code="72148",
        diagnosis_description="Lumbar radiculopathy",
        diagnosis_code="M54.16",
        clinical_justification="Failed 6 weeks of PT, positive SLR.",
        extraction_confidence=0.95,
    )
    chunk = PolicyChunk(
        source_document="lumbar_mri_NIA_CG_044.txt",
        source_title="Lumbar Spine MRI Clinical Guidelines (NIA_CG_044)",
        section="INDICATIONS FOR LUMBAR SPINE MRI",
        subsection="For Evaluation of Back Pain",
        chunk_text=(
            "Lumbar spine MRI is indicated for back pain with any of the following:\n"
            "- Failure of conservative treatment for at least six (6) weeks\n"
            "- New or worsening objective neurologic deficits on exam"
        ),
        relevance_score=0.77,
    )
    # Fabricate a proposer output with an invented citation
    fabricated_proposal = ProposerDecision(
        proposed_decision="approve",
        rationale="Policy explicitly states MRI is required for all radiculopathy cases.",
        criteria_met=["Policy mandates MRI for any radiculopathy diagnosis"],
        criteria_unmet=[],
        policy_citations=[
            "MRI is mandatory for all cases of lumbar radiculopathy regardless of treatment history"
        ],
        confidence=0.9,
    )
    state = {
        "case_id": "test_fabricated",
        "raw_request_text": "",
        "extracted_request": extracted,
        "extraction_flags": [],
        "retrieved_policy_chunks": [chunk],
        "retrieval_flags": [],
        "proposer_decision": fabricated_proposal,
        "proposer_flags": [],
    }
    result = critique_proposal(state)
    feedback = result["critic_feedback"]

    # The fabricated citation should be challenged
    assert len(feedback.challenged_citations) > 0, (
        "Expected critic to challenge the fabricated citation 'MRI is mandatory for all cases...'"
    )
