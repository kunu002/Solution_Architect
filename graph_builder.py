# graph_builder.py
from langgraph.graph import StateGraph, START, END
from state import ChatState
from agents.solution_agent import solution_agent
from agents.architect_agent import architect_agent
from agents.analysis_agent import analysis_agent
from agents.supervisor_agent import supervisor_agent

def _route_from_supervisor(state: ChatState) -> str:
    """
    Supervisor writes state['route'] as:
    'solution_agent' | 'architect_agent' | 'analysis_agent' | 'END' | None
    """
    route = state.get("route")
    if route is None:
        return "supervisor"   # stay on supervisor until we have a decision
    return route

def build_graph():
    """
    Builds the LangGraph workflow with supervisor and worker agents.
    Ensures that control returns to the supervisor after each worker agent's execution.
    """
    g = StateGraph(ChatState)

    g.add_node("supervisor", supervisor_agent)
    g.add_node("solution_agent", solution_agent)
    g.add_node("architect_agent", architect_agent)
    g.add_node("analysis_agent", analysis_agent)

    # Initial entry point into the graph is the supervisor
    g.add_edge(START, "supervisor")

    # Conditional edges from the supervisor to decide the next agent or end the flow
    g.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "solution_agent": "solution_agent",
            "architect_agent": "architect_agent",
            "analysis_agent": "analysis_agent",
            "END": END,
            # If supervisor returns None or an unknown route, loop back to itself
            "supervisor": "supervisor",
        },
    )

    # After any worker agent completes, return control to the supervisor
    g.add_edge("solution_agent", "supervisor")
    g.add_edge("architect_agent", "supervisor")
    g.add_edge("analysis_agent", "supervisor")

    return g.compile()
