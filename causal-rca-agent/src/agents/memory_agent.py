"""
Memory Agent — Long-term memory with pipeline short-circuit

Agentic practices demonstrated:
  - Long-term memory: ChromaDB stores and retrieves past incident signatures
  - Pipeline short-circuit: high-confidence match skips full causal reasoning
  - Parallel execution: runs alongside Evidence Collector
"""

from typing import Any, Dict

from src.tools.memory_tools import search_memory
from src.utils.logging import get_logger
from src.utils.state import RCAState

log = get_logger("memory_agent")


def node_search_memory(state: RCAState) -> Dict[str, Any]:
    """
    Embeds the incident and searches ChromaDB for past matches.
    If similarity >= threshold the Supervisor will fast-path
    to verification instead of full causal discovery.
    """
    evidence = state.get("evidence", {})
    if not evidence:
        log.info("memory_skip", reason="no_evidence_yet")
        return {"memory_match": None, "steps": ["🧠 Memory Agent: no evidence available yet"]}

    log.info("memory_search_start")
    match = search_memory(evidence)

    if match:
        log.info(
            "memory_match_found",
            incident_id=match["incident_id"],
            similarity=match["similarity"],
        )
        return {
            "memory_match": match,
            "steps": [
                f"🧠 Memory Agent: matched incident {match['incident_id']} "
                f"(similarity {match['similarity']}) — "
                f"past resolution: {match['resolution'][:80]}"
            ],
        }

    log.info("memory_no_match")
    return {
        "memory_match": None,
        "steps": ["🧠 Memory Agent: no match above threshold — full causal discovery required"],
    }
