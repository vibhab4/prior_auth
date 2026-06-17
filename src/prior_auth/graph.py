"""Compile the LangGraph StateGraph for the prior authorization pipeline.

Call `build_graph()` to get a runnable graph. As nodes are added in later
increments, new `add_node` / `add_edge` calls go here. Conditional branching
(e.g. routing low-confidence cases to human review) arrives when the judge
node (Node 5) is built.
"""

from langgraph.graph import END, START, StateGraph

from prior_auth.nodes.extract import extract_request
from prior_auth.nodes.retrieve_policy import retrieve_policy
from prior_auth.state import PriorAuthState


def build_graph():
    """Build and compile the prior-auth review graph.

    Current topology (Phase 1, Increments 1-2):
        START -> extract_request -> retrieve_policy -> END

    Returns a compiled LangGraph graph that accepts PriorAuthState-shaped
    dicts via .invoke() / .stream().
    """
    graph = StateGraph(PriorAuthState)

    graph.add_node("extract_request", extract_request)
    graph.add_node("retrieve_policy", retrieve_policy)

    graph.add_edge(START, "extract_request")
    graph.add_edge("extract_request", "retrieve_policy")
    graph.add_edge("retrieve_policy", END)

    return graph.compile()
