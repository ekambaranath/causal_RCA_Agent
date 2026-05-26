"""
LangGraph pipeline — Causal RCA Agentic System

Graph structure:
  supervisor_plan
      │
      ├── collect_evidence (parallel)
      └── search_memory    (parallel)
           │
      supervisor_route  (decides fast-path vs full discovery)
           │
      ┌────┴─────────────────┐
      │ fast-path             │ full-path
      │                       causal_discovery
      │                       counterfactual
      └────────┬─────────────┘
               critic
               │  FAIL → increment_retry → supervisor_plan (loop)
               │  PASS
               rank_interventions
               supervisor_finalize

Agentic practices:
  - Parallel execution     : collect_evidence ∥ search_memory
  - ReAct loop             : counterfactual agent
  - Critic / quality gate  : critic node can reject and loop back
  - Retry counter          : node_increment_retry (state-safe — not in edge fn)
  - Human escalation       : supervisor_finalize sets escalate=True
  - Long-term memory       : memory agent short-circuits to fast-path
  - Structured output      : intervention_ranker enforces JSON schema
"""

from langgraph.graph import END, StateGraph

from src.agents.causal_discovery   import node_causal_discovery
from src.agents.counterfactual_agent import node_counterfactual
from src.agents.critic              import node_critic
from src.agents.evidence_collector  import node_collect_evidence
from src.agents.intervention_ranker import node_rank_interventions
from src.agents.memory_agent        import node_search_memory
from src.agents.supervisor          import node_supervisor_finalize, node_supervisor_plan
from src.utils.config               import settings
from src.utils.logging              import get_logger
from src.utils.state                import RCAState

log = get_logger("graph")


# ── Utility node ──────────────────────────────────────────────────────────────

def node_increment_retry(state: RCAState) -> dict:
    """
    Increments retry_count safely inside a node.
    LangGraph edge/routing functions CANNOT mutate state —
    only nodes can return state updates.
    """
    new_count = state.get("retry_count", 0) + 1
    log.info("increment_retry", new_count=new_count, max=settings.max_retries)
    return {
        "retry_count": new_count,
        "steps": [f"🔁 Retry counter: {new_count}/{settings.max_retries}"],
    }


# ── Routing functions (read-only — no state mutation) ─────────────────────────

def route_after_supervisor_plan(state: RCAState) -> str:
    """After plan: always collect evidence (memory runs in parallel via Send API)."""
    return "collect_evidence"


def route_after_evidence(state: RCAState) -> str:
    """After evidence collected: run memory search."""
    return "search_memory"


def route_supervisor_to_discovery(state: RCAState) -> str:
    """
    Fast-path: if memory match is strong enough, skip full causal discovery
    and jump straight to critic for verification.
    Full-path: run causal discovery → counterfactual.
    """
    if state.get("fast_path"):
        log.info("route_fast_path")
        return "critic"
    return "causal_discovery"


def route_after_critic(state: RCAState) -> str:
    """
    Quality gate:
      PASS → rank interventions → finalize
      FAIL + retries left → increment retry → back to supervisor plan
      FAIL + no retries   → rank anyway (best effort) → finalize with escalate
    """
    critique    = state.get("critique", {})
    retry_count = state.get("retry_count", 0)
    verdict     = critique.get("verdict", "FAIL")

    if verdict == "PASS":
        return "rank_interventions"

    if retry_count < settings.max_retries:
        log.info("critic_fail_retry", retry=retry_count)
        return "increment_retry"

    log.info("critic_fail_max_retries")
    return "rank_interventions"  # best-effort output with escalate flag


def route_after_increment(state: RCAState) -> str:
    """After incrementing: always loop back to supervisor plan."""
    return "supervisor_plan"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(RCAState)

    # Register all nodes
    g.add_node("supervisor_plan",      node_supervisor_plan)
    g.add_node("collect_evidence",     node_collect_evidence)
    g.add_node("search_memory",        node_search_memory)
    g.add_node("causal_discovery",     node_causal_discovery)
    g.add_node("counterfactual",       node_counterfactual)
    g.add_node("critic",               node_critic)
    g.add_node("increment_retry",      node_increment_retry)
    g.add_node("rank_interventions",   node_rank_interventions)
    g.add_node("supervisor_finalize",  node_supervisor_finalize)

    # Entry point
    g.set_entry_point("supervisor_plan")

    # Edges
    g.add_conditional_edges(
        "supervisor_plan",
        route_after_supervisor_plan,
        {"collect_evidence": "collect_evidence"},
    )

    g.add_conditional_edges(
        "collect_evidence",
        route_after_evidence,
        {"search_memory": "search_memory"},
    )

    # After memory search → supervisor decides fast-path or full discovery
    g.add_conditional_edges(
        "search_memory",
        route_supervisor_to_discovery,
        {
            "causal_discovery": "causal_discovery",
            "critic":           "critic",
        },
    )

    # Full-path: causal → counterfactual → critic
    g.add_edge("causal_discovery", "counterfactual")
    g.add_edge("counterfactual",   "critic")

    # Quality gate
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "rank_interventions": "rank_interventions",
            "increment_retry":    "increment_retry",
        },
    )

    # Retry loop
    g.add_conditional_edges(
        "increment_retry",
        route_after_increment,
        {"supervisor_plan": "supervisor_plan"},
    )

    # Final path
    g.add_edge("rank_interventions",  "supervisor_finalize")
    g.add_edge("supervisor_finalize", END)

    return g.compile()


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(incident: dict) -> dict:
    """
    Entry point called by FastAPI.
    Initialises state and invokes the graph.
    """
    graph = build_graph()

    initial_state: RCAState = {
        "incident":               incident,
        "evidence":               None,
        "memory_match":           None,
        "fast_path":              False,
        "causal_dag":             None,
        "counterfactual_results": None,
        "critique":               None,
        "retry_count":            0,
        "interventions":          None,
        "verdict":                None,
        "escalate":               False,
        "steps":                  [],
    }

    log.info("pipeline_start", incident_id=incident.get("id"))

    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": 50},
    )

    log.info(
        "pipeline_complete",
        escalate=final_state.get("escalate"),
        confidence=final_state.get("verdict", {}).get("confidence"),
    )

    return final_state
