from typing import NotRequired, TypedDict

from prior_auth.schemas.extraction import ExtractedRequest
from prior_auth.schemas.policy import PolicyChunk


class PriorAuthState(TypedDict):
    """Shared state passed between LangGraph nodes.

    Only fields used by nodes that exist so far are defined as real keys.
    Fields below are commented as a map of the full Phase 1 data flow --
    they're added (with their own schemas) as each node is built.
    """

    # --- Input ---
    case_id: str
    raw_request_text: str

    # --- Written by: extract_request (Node 1) ---
    extracted_request: NotRequired[ExtractedRequest]
    extraction_flags: NotRequired[list[str]]

    # --- Written by: retrieve_policy (Node 2) ---
    retrieved_policy_chunks: NotRequired[list[PolicyChunk]]
    retrieval_flags: NotRequired[list[str]]

    # --- Future fields, added incrementally ---
    # proposer_decision         -> proposer (Node 3)
    # critic_feedback           -> critic (Node 4)
    # final_decision, final_rationale, confidence, requires_human_review
    #                           -> judge (Node 5)
