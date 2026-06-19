"""SQLite-backed checkpointer for LangGraph state persistence.

The checkpointer saves the full graph state (all node outputs) to a SQLite
database after each node completes. This enables two things:

1. Resuming "need more info" cases: when a case is flagged and the provider
   later submits additional documentation, the saved state is loaded and the
   graph is re-run with the new information -- no need to re-extract or
   re-retrieve from scratch.

2. Audit trail: every node's output for every run is persisted, which
   supports the HIPAA-aware design goal of keeping a full record of how
   each decision was reached.

HIPAA note: in a production system this database would contain patient-adjacent
data and would need to be encrypted at rest and access-controlled. For this
portfolio project the data is entirely synthetic ([SYNTHETIC] labeled).
"""

from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

# Default path: data/checkpoints.db, gitignored as a derived artifact.
CHECKPOINT_DB_PATH = Path(__file__).parents[2] / "data" / "checkpoints.db"


@contextmanager
def get_checkpointer(db_path: Path = CHECKPOINT_DB_PATH):
    """Context manager that yields a SqliteSaver for the given database path.

    Usage:
        with get_checkpointer() as checkpointer:
            app = build_graph(checkpointer=checkpointer)
            result = app.invoke(state, config={"configurable": {"thread_id": "case_001"}})

    SqliteSaver must be used as a context manager to properly open and close
    the underlying SQLite connection. Using it outside a `with` block will
    raise a connection error.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        yield checkpointer
