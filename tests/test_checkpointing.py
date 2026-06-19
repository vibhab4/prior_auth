"""Tests for Phase 2: SQLite checkpointing.

These tests verify that:
1. build_graph() with a checkpointer saves state to SQLite
2. State can be retrieved after a run completes
3. All node outputs are present in the saved state
4. A follow-up run produces a different (better) result than the initial run

These tests make real Anthropic API calls and require ANTHROPIC_API_KEY.
They use an in-memory SQLite database (:memory:) so no files are written to disk.

Run with: pytest tests/test_checkpointing.py -v
"""

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from prior_auth.graph import build_graph


CASE_002_TEXT = """\
PRIOR AUTHORIZATION REQUEST

Patient: [SYNTHETIC] Morgan P., DOB 09/22/1965 (Age 60), Sex: F
Requesting Provider: Dr. L. Chen, Sports Medicine Clinic
Requested Service: Right knee arthroscopy with partial meniscectomy, outpatient
Requested CPT: 29881 (Arthroscopy, knee, surgical; with meniscectomy)
Diagnosis: Medial meniscus tear, right knee, with mechanical symptoms.

Clinical Notes:
Patient reports 5 months of right knee pain and intermittent locking.
MRI 03/2026 shows complex medial meniscus tear with displaced fragment.
Patient completed a course of physical therapy and a corticosteroid injection
with only temporary relief. Positive McMurray test.
"""

CASE_002_FOLLOWUP_TEXT = """\
PRIOR AUTHORIZATION REQUEST — ADDENDUM

Patient: [SYNTHETIC] Morgan P., DOB 09/22/1965 (Age 60), Sex: F
Requesting Service: Right knee arthroscopy with partial meniscectomy, outpatient
Requested CPT: 29881
ICD-10: M23.201 (Derangement of medial meniscus due to old tear, right knee)
Diagnosis: Medial meniscus tear, right knee, with mechanical symptoms.

Additional documentation:
- OA grading: Kellgren-Lawrence Grade I. Outerbridge Grade I-II.
- PT records: 8 weeks formal in-person PT (04-05/2026), documented.
- MRI (03/2026): Bucket-handle medial meniscus tear, displaced fragment,
  no significant chondral damage.

Clinical Notes:
Patient reports 5 months of right knee pain and intermittent locking.
MRI confirms complex tear with displaced bucket-handle fragment.
Completed 8 weeks formal PT and corticosteroid injection with only temporary
relief. Positive McMurray test. Mild effusion.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_state_is_persisted_after_run():
    """After a run, app.get_state() should return the complete final state."""
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test_persist"}}

        app.invoke(
            {"case_id": "test_persist", "raw_request_text": CASE_002_TEXT},
            config=config,
        )

        saved = app.get_state(config)
        assert saved is not None
        assert saved.values.get("final_decision") is not None
        assert saved.values.get("extracted_request") is not None
        assert saved.values.get("retrieved_policy_chunks") is not None


def test_all_node_outputs_in_checkpoint():
    """The saved checkpoint must contain outputs from all 5 nodes."""
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test_all_nodes"}}

        app.invoke(
            {"case_id": "test_all_nodes", "raw_request_text": CASE_002_TEXT},
            config=config,
        )

        saved = app.get_state(config)
        values = saved.values

        # Every node's output should be in the checkpoint
        assert "extracted_request" in values       # Node 1
        assert "retrieved_policy_chunks" in values  # Node 2
        assert "proposer_decision" in values        # Node 3
        assert "critic_feedback" in values          # Node 4
        assert "final_decision" in values           # Node 5


def test_graph_completed_no_pending_nodes():
    """After .invoke() completes, there should be no next nodes to run."""
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test_completed"}}

        app.invoke(
            {"case_id": "test_completed", "raw_request_text": CASE_002_TEXT},
            config=config,
        )

        saved = app.get_state(config)
        # Empty tuple means the graph has finished -- no nodes waiting to run
        assert saved.next == ()


def test_followup_run_improves_decision():
    """The follow-up submission (with complete docs) should produce a better
    outcome than the initial submission (missing ICD code, OA grading)."""
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = build_graph(checkpointer=checkpointer)

        # Initial run
        config_initial = {"configurable": {"thread_id": "case_002_initial"}}
        state_initial = app.invoke(
            {"case_id": "case_002_initial", "raw_request_text": CASE_002_TEXT},
            config=config_initial,
        )
        final_initial = state_initial["final_decision"]

        # Follow-up run
        config_followup = {"configurable": {"thread_id": "case_002_followup"}}
        state_followup = app.invoke(
            {"case_id": "case_002_followup", "raw_request_text": CASE_002_FOLLOWUP_TEXT},
            config=config_followup,
        )
        final_followup = state_followup["final_decision"]

        # The initial run should have more uncertainty or need-more-info
        # The follow-up with complete docs should have higher confidence
        assert final_followup.confidence >= final_initial.confidence, (
            f"Follow-up confidence ({final_followup.confidence}) should be >= "
            f"initial ({final_initial.confidence})"
        )

        # The follow-up should not be a denial (it has all required info)
        assert final_followup.final_decision != "deny", (
            f"Follow-up with complete documentation should not be denied. "
            f"Got: {final_followup.final_decision}"
        )


def test_different_thread_ids_are_independent():
    """Two runs with different thread_ids must not interfere with each other."""
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = build_graph(checkpointer=checkpointer)

        config_a = {"configurable": {"thread_id": "independent_a"}}
        config_b = {"configurable": {"thread_id": "independent_b"}}

        app.invoke(
            {"case_id": "independent_a", "raw_request_text": CASE_002_TEXT},
            config=config_a,
        )
        app.invoke(
            {"case_id": "independent_b", "raw_request_text": CASE_002_FOLLOWUP_TEXT},
            config=config_b,
        )

        saved_a = app.get_state(config_a)
        saved_b = app.get_state(config_b)

        # Each thread should have its own case_id
        assert saved_a.values["case_id"] == "independent_a"
        assert saved_b.values["case_id"] == "independent_b"
