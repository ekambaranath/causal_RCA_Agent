import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class RCAState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    incident: Dict[str, Any]           # Raw incident payload from API

    # ── Evidence phase ────────────────────────────────────────────────────────
    evidence: Optional[Dict[str, Any]] # Events, metrics, deploy history
    memory_match: Optional[Dict[str, Any]]  # Past incident match from ChromaDB
    fast_path: bool                    # True → memory match found, skip full discovery

    # ── Causal phase ──────────────────────────────────────────────────────────
    causal_dag: Optional[Dict[str, Any]]          # Nodes + edges from causal algo
    counterfactual_results: Optional[List[Dict]]  # Results per tested candidate

    # ── Quality gate ──────────────────────────────────────────────────────────
    critique: Optional[Dict[str, Any]]  # Critic agent verdict
    retry_count: int                    # Incremented by node_increment_retry

    # ── Output ────────────────────────────────────────────────────────────────
    interventions: Optional[List[Dict]]   # Ranked fix list
    verdict: Optional[Dict[str, Any]]     # Final root cause + confidence
    escalate: bool                        # True → hand off to human

    # ── Audit trail (appended by every node) ──────────────────────────────────
    steps: Annotated[List[str], operator.add]
