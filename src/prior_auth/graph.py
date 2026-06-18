"""Compile the LangGraph StateGraph for the prior authorization pipeline.

Call `build_graph()` to get a runnable graph. As nodes are added in later
increments, new `add_node` / `add_edge` calls go here. Conditional branching
(e.g. routing low-confidence cases to human review) arrives when the judge
node (Node 5) is built.
"""

from langgraph.graph import END, START, StateGraph

from prior_auth.nodes.critique import critique_proposal
from prior_auth.nodes.extract import extract_request
from prior_auth.nodes.judge import judge_decision
from prior_auth.nodes.propose import propose_coverage
from prior_auth.nodes.retrieve_policy import retrieve_policy
from prior_auth.state import PriorAuthState


def build_graph():
    """Build and compile the prior-auth review graph.

    Phase 1 complete topology (Increments 1-5):
        START -> extract_request -> retrieve_policy -> propose_coverage
             -> critique_proposal -> judge_decision -> END

    Returns a compiled LangGraph graph that accepts PriorAuthState-shaped
    dicts via .invoke() / .stream().
    """
    graph = StateGraph(PriorAuthState)

    graph.add_node("extract_request", extract_request)
    graph.add_node("retrieve_policy", retrieve_policy)
    graph.add_node("propose_coverage", propose_coverage)
    graph.add_node("critique_proposal", critique_proposal)
    graph.add_node("judge_decision", judge_decision)

    graph.add_edge(START, "extract_request")
    graph.add_edge("extract_request", "retrieve_policy")
    graph.add_edge("retrieve_policy", "propose_coverage")
    graph.add_edge("propose_coverage", "critique_proposal")
    graph.add_edge("critique_proposal", "judge_decision")
    graph.add_edge("judge_decision", END)

    return graph.compile()
