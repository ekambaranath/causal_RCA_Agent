"""
Evidence Collector Agent — Tool-use agent

Agentic practices demonstrated:
  - Tool use: autonomously decides which telemetry tools to call
  - Parallel execution: runs alongside Memory Agent (LangGraph parallel nodes)
"""

from typing import Any, Dict

from src.tools.telemetry_tools import gather_telemetry
from src.utils.logging import get_logger
from src.utils.state import RCAState

log = get_logger("evidence_collector")


def node_collect_evidence(state: RCAState) -> Dict[str, Any]:
    """
    Calls telemetry tools to gather the raw evidence packet.
    In production: calls OpenTelemetry, Prometheus, k8s events API.
    """
    incident = state["incident"]
    log.info("evidence_collect", incident_id=incident.get("id"))

    evidence = gather_telemetry(incident)

    log.info(
        "evidence_collected",
        events=len(evidence.get("events", [])),
        services=evidence.get("services"),
        deploys=len(evidence.get("deploys", [])),
    )

    return {
        "evidence": evidence,
        "steps": [
            f"📡 Evidence Collector: gathered {len(evidence['events'])} events "
            f"from {len(evidence['services'])} services over "
            f"{evidence['window_minutes']} min window"
        ],
    }
