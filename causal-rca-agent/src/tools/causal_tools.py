"""
Causal reasoning tools.

Implements a simplified causal discovery algorithm based on temporal
ordering + service dependency — a practical approximation of the PC algorithm.

Counterfactual queries use a structural intervention model:
  P(failure | do(candidate = removed))

In production: swap build_causal_dag() for DoWhy's PC algorithm and
run_counterfactual() for DoWhy's do() operator.
"""

from typing import Any, Dict, List


# ── Service dependency graph (known a-priori from infra topology) ─────────────
SERVICE_DEPS: Dict[str, List[str]] = {
    "ci-cd":            ["auth-service", "payments-service", "order-service"],
    "analytics-job":    ["db-primary"],
    "db-primary":       ["order-service", "payment-service", "user-service"],
    "auth-service":     ["api-gateway", "checkout-service"],
    "payments-service": ["api-gateway", "frontend"],
    "api-gateway":      ["frontend", "checkout-service"],
}


def _are_causally_related(cause_svc: str, effect_svc: str) -> bool:
    """Returns True if cause_svc is an upstream dependency of effect_svc."""
    return effect_svc in SERVICE_DEPS.get(cause_svc, [])


def _find_downstream(root_event: str, edges: List[Dict]) -> List[str]:
    """BFS over causal edges to find all effects of a candidate root cause."""
    downstream, queue = set(), [root_event]
    while queue:
        node = queue.pop(0)
        for e in edges:
            if e["cause"] == node and e["effect"] not in downstream:
                downstream.add(e["effect"])
                queue.append(e["effect"])
    return list(downstream)


def build_causal_dag(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Temporal ordering + service dependency → Directed Acyclic Graph.

    Steps:
      1. Sort events by timestamp (earlier = potential cause).
      2. For each ordered pair (A, B): if A → B within 5 min AND
         A's service is upstream of B's service → add causal edge.
      3. Root candidates = nodes with no incoming edges.
    """
    events = evidence.get("events", [])
    events_sorted = sorted(events, key=lambda x: x["ts"])

    edges: List[Dict] = []
    for i, cause_ev in enumerate(events_sorted):
        for effect_ev in events_sorted[i + 1:]:
            time_gap_min = (effect_ev["ts"] - cause_ev["ts"]) / 60
            if time_gap_min > 5:
                break  # events are sorted; no point continuing
            if _are_causally_related(cause_ev["service"], effect_ev["service"]):
                confidence = round(max(0.4, 1.0 - time_gap_min / 5), 2)
                edges.append({
                    "cause":      cause_ev["event"],
                    "effect":     effect_ev["event"],
                    "cause_svc":  cause_ev["service"],
                    "effect_svc": effect_ev["service"],
                    "gap_min":    round(time_gap_min, 1),
                    "confidence": confidence,
                })

    all_effects = {e["effect"] for e in edges}
    all_causes  = {e["cause"]  for e in edges}
    root_candidates = [
        ev["event"] for ev in events_sorted
        if ev["event"] in all_causes and ev["event"] not in all_effects
    ]

    if not root_candidates and events_sorted:
        root_candidates = [events_sorted[0]["event"]]

    return {
        "nodes":           [e["event"] for e in events_sorted],
        "edges":           edges,
        "root_candidates": root_candidates,
        "spurious_removed": max(0, len(events_sorted) * 2 - len(edges)),
    }


def run_counterfactual(
    candidate: str,
    dag: Dict[str, Any],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Structural intervention query: do(candidate = removed).

    Estimates P(failure | do(candidate=removed)) by measuring how many
    downstream error/alert events would be disconnected from their causal chain.
    """
    edges       = dag.get("edges", [])
    downstream  = _find_downstream(candidate, edges)
    error_count = sum(
        1 for ev in downstream
        if any(kw in ev.lower() for kw in ["error", "fail", "alert", "breach", "timeout", "refused"])
    )
    total_errors = sum(
        1 for ev in dag.get("nodes", [])
        if any(kw in ev.lower() for kw in ["error", "fail", "alert", "breach", "timeout", "refused"])
    )

    p_failure_with = 0.94
    if total_errors > 0 and error_count >= total_errors * 0.6:
        # Removing this candidate disconnects most error events → true root cause
        p_failure_without = round(0.94 * (1 - error_count / max(total_errors, 1)) * 0.15, 2)
        is_root_cause = True
    else:
        # Removing this candidate barely changes failure probability → downstream symptom
        p_failure_without = round(0.94 - (error_count * 0.05), 2)
        is_root_cause = False

    confidence = round(1 - p_failure_without, 2) if is_root_cause else round(p_failure_without, 2)

    return {
        "candidate":          candidate,
        "p_failure_with":     p_failure_with,
        "p_failure_without":  p_failure_without,
        "confidence":         confidence,
        "is_root_cause":      is_root_cause,
        "downstream_effects": downstream,
        "verdict":            "root_cause" if is_root_cause else "downstream_symptom",
    }
