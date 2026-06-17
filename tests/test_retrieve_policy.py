"""Tests for the retrieve_policy node.

These tests are deterministic -- they do NOT call the Anthropic API.
Retrieval uses only the local bge-small embedding model against the
pre-built Chroma store.

Prerequisite: run `python scripts/build_index.py` before running these tests.
The vectorstore must exist at data/vectorstore/.

Observed relevance scores (bge-small-en-v1.5, cosine similarity):
  - On-target queries (case_001 lumbar, case_002 knee): top score 0.76-0.78
  - Off-target / vague queries (case_003 shoulder): top score ~0.60
  MIN_RELEVANCE_SCORE=0.65 cleanly separates the two groups.
"""

from prior_auth.nodes.retrieve_policy import retrieve_policy
from prior_auth.schemas.extraction import ExtractedRequest
from prior_auth.schemas.policy import PolicyChunk


def _make_state(extracted: ExtractedRequest) -> dict:
    return {
        "case_id": "test",
        "raw_request_text": "",
        "extracted_request": extracted,
        "extraction_flags": [],
    }


def test_lumbar_mri_retrieves_correct_policy():
    """Lumbar MRI for radiculopathy after failed PT should match lumbar MRI corpus."""
    extracted = ExtractedRequest(
        procedure_description="Lumbar MRI without contrast",
        procedure_code="72148",
        diagnosis_description="Lumbar radiculopathy, suspected L5-S1 disc herniation",
        diagnosis_code="M54.16",
        clinical_justification=(
            "Patient has 14 weeks of low back pain radiating to left leg, positive "
            "straight leg raise, failed 6 weeks of physical therapy and NSAIDs."
        ),
        extraction_confidence=0.99,
    )
    result = retrieve_policy(_make_state(extracted))
    chunks = result["retrieved_policy_chunks"]

    assert len(chunks) > 0
    assert all(isinstance(c, PolicyChunk) for c in chunks)
    assert chunks[0].relevance_score > 0.65, (
        f"Expected top score > 0.65, got {chunks[0].relevance_score}"
    )
    assert "lumbar" in chunks[0].source_document.lower()
    assert "no_relevant_policy_found" not in result["retrieval_flags"]


def test_knee_arthroscopy_retrieves_correct_policy():
    """Knee arthroscopy for meniscal tear should match knee arthroscopy corpus."""
    extracted = ExtractedRequest(
        procedure_description="Right knee arthroscopy with partial meniscectomy",
        procedure_code="29881",
        diagnosis_description="Medial meniscus tear, right knee, with mechanical symptoms",
        diagnosis_code=None,
        clinical_justification=(
            "Patient has knee pain with locking and giving way, MRI confirms medial "
            "meniscus tear. Failed 8 weeks of physical therapy and steroid injection."
        ),
        extraction_confidence=0.97,
    )
    result = retrieve_policy(_make_state(extracted))
    chunks = result["retrieved_policy_chunks"]

    assert len(chunks) > 0
    assert chunks[0].relevance_score > 0.65
    source_docs = [c.source_document for c in chunks]
    assert any("knee" in d.lower() for d in source_docs)
    assert "no_relevant_policy_found" not in result["retrieval_flags"]


def test_sparse_vague_request_flags_no_relevant_policy():
    """Vague 'imaging study for shoulder pain' has no matching policy in the corpus.

    The corpus covers lumbar MRI, knee arthroscopy, and general MRI (NCD 220.2).
    A shoulder-specific query scores noticeably lower (~0.60) than an on-target
    query (~0.76), falling below MIN_RELEVANCE_SCORE=0.65.
    """
    extracted = ExtractedRequest(
        procedure_description="Imaging study for shoulder pain",
        procedure_code=None,
        diagnosis_description="Shoulder pain",
        diagnosis_code=None,
        # Realistic justification matching actual case_003 extraction output.
        # More text drives the embedding toward lower similarity against the
        # lumbar/knee/general-MRI corpus (observed top score: ~0.60).
        clinical_justification=(
            "The provider notes the patient has had shoulder pain for an unspecified "
            "duration and is requesting imaging to further evaluate the cause. No "
            "specific symptoms, severity, duration, failed conservative treatments, "
            "or relevant exam findings are documented in the referral note."
        ),
        extraction_confidence=0.25,
    )
    result = retrieve_policy(_make_state(extracted))

    # Should still return chunks (never empty), but flag the low relevance.
    assert len(result["retrieved_policy_chunks"]) > 0
    assert "no_relevant_policy_found" in result["retrieval_flags"]


def test_chunks_carry_citation_metadata():
    """Every returned chunk must have the metadata fields needed for auditable citation."""
    extracted = ExtractedRequest(
        procedure_description="Lumbar MRI without contrast",
        procedure_code="72148",
        diagnosis_description="Lumbar radiculopathy",
        diagnosis_code="M54.16",
        clinical_justification="Failed conservative therapy, positive SLR.",
        extraction_confidence=0.95,
    )
    result = retrieve_policy(_make_state(extracted))

    for chunk in result["retrieved_policy_chunks"]:
        assert chunk.source_document, "source_document must be non-empty"
        assert chunk.source_title, "source_title must be non-empty"
        assert chunk.chunk_text, "chunk_text must be non-empty"
        assert chunk.relevance_score is not None
        # section may be None for file-header chunks, but score should be valid
        assert 0.0 <= chunk.relevance_score <= 1.0
