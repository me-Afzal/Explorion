"""
graph.py  (assistant package)
------------------------------
LangGraph pipeline — no end_node; graph returns final state directly.

Topology:
                   ┌──────────────────────┐
    START ──────►  │     manager_node     │
                   └──────────┬───────────┘
                               │
               ┌───────────────┴──────────────┐
         ready=True                      ready=False
               │                              │
  ┌────────────▼────────────┐               END
  │     research_node       │
  └────────────┬────────────┘
  ┌────────────▼────────────┐
  │      analyze_node       │
  └────────────┬────────────┘
              END
"""

from typing import TypedDict, Optional, Literal

from langgraph.graph import StateGraph, END
from langsmith import traceable

from .agents import manager_agent, research_agent, analyze_agent


# ══════════════════════════════════════════════════════════════════════════════
# STATE SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class PipelineState(TypedDict, total=False):
    user_input:        str
    context:           dict   # topic / goal / location
    session_id:        str
    buffer_memory:     list   # last 2 user/assistant turns
    retrieved_context: str    # VectorDB top-5 chunks
    manager_response:  str
    ready_to_research: bool
    research_findings: str
    final_report:      str


# ══════════════════════════════════════════════════════════════════════════════
# NODES
# ══════════════════════════════════════════════════════════════════════════════

@traceable(name="manager_node")
def manager_node(state: PipelineState) -> PipelineState:
    result = manager_agent(state)
    if result["ready_to_research"]:
        print("[System: routing to deep research]")
    else:
        print("[System: manager response — no research triggered]")
    return {
        "context":           result["context"],
        "manager_response":  result["manager_response"],
        "ready_to_research": result["ready_to_research"],
    }


@traceable(name="research_node")
def research_node(state: PipelineState) -> PipelineState:
    print("\n[System: Research Agent — searching the web…]\n")
    result = research_agent(state)
    return {"research_findings": result["research_findings"]}


@traceable(name="analyze_node")
def analyze_node(state: PipelineState) -> PipelineState:
    print("\n[System: Analyze Agent — generating report…]\n")
    result = analyze_agent(state)
    return {"final_report": result["final_report"]}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════════

def route_after_manager(state: PipelineState) -> Literal["research_node", "end"]:
    if state.get("ready_to_research"):
        return "research_node"
    return "end"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    builder = StateGraph(PipelineState)

    builder.add_node("manager_node",  manager_node)
    builder.add_node("research_node", research_node)
    builder.add_node("analyze_node",  analyze_node)

    builder.set_entry_point("manager_node")

    builder.add_conditional_edges(
        "manager_node",
        route_after_manager,
        {
            "research_node": "research_node",
            "end":           END,
        },
    )

    builder.add_edge("research_node", "analyze_node")
    builder.add_edge("analyze_node",  END)

    return builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════

def stream_pipeline(graph: StateGraph, state: PipelineState):
    """
    Generator entry point (Streamlit UI).
    Yields (node_name, state_update) as each node completes so the UI can
    show live progress. Wrap the call site with langsmith.trace() for tracing.
    """
    for chunk in graph.stream(state, stream_mode="updates"):
        for node_name, update in chunk.items():
            yield node_name, update
