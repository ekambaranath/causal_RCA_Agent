"""
Supervisor Agent — Orchestrator

Agentic practices demonstrated:
  - Planning: forms an initial hypothesis from the incident
  - ReAct routing: decides which agents to call and in what order
  - Escalation: hands off to human when confidence is too low after retries
"""

from typing import Any, Dict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from src.utils.config import settings
from src.utils.logging import get_logger
from src.utils.state import RCAState

log = get_logger("supervisor")


def _build_llm() -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model_large,
        num_predict=256,
        temperature=settings.temperature,
        timeout=settings.llm_timeout,
    )


def node_supervisor_plan(state: RCAState) -> Dict[str, Any]:
    """
    Entry node. Reads the incident and forms an initial hypothesis.
    Sets fast_path=True if memory already returned a high-confidence match.
    """
    incident = state["incident"]
    memory   = state.get("memory_match")

    log.info("supervisor_plan", incident_id=incident.get("id"), memory_match=bool(memory))

    llm    = _build_llm()
    prompt = (
        f"You are an SRE supervisor. A new incident has been reported.\n\n"
        f"Incident: {incident.get('description', 'Unknown incident')}\n"
        f"Service: {incident.get('service', 'Unknown service')}\n"
        f"Error rate: {incident.get('error_rate', 'Unknown')}%\n\n"
    )

    if memory:
        prompt += (
            f"Memory match found (similarity {memory['similarity']}):\n"
            f"Past root cause: {memory['root_cause']}\n"
            f"Past resolution: {memory['resolution']}\n\n"
            f"Given this strong match, briefly confirm the investigation plan "
            f"and whether the past resolution is likely to apply."
        )
    else:
        prompt += (
            "No memory match found. Form an initial hypothesis about the most "
            "likely root cause category (deployment/database/memory/network) "
            "and which signals to focus on. Be concise — 2-3 sentences."
        )

    response = llm.invoke([HumanMessage(content=prompt)])
    plan     = response.content.strip()

    fast_path = bool(memory and memory["similarity"] >= settings.memory_similarity_threshold)

    log.info("supervisor_plan_formed", fast_path=fast_path, plan_preview=plan[:80])

    return {
        "fast_path": fast_path,
        "steps": [f"🧠 Supervisor: {plan}"],
    }


def node_supervisor_finalize(state: RCAState) -> Dict[str, Any]:
    """
    Final node. Aggregates all agent outputs into the definitive verdict.
    Decides whether to escalate to a human.
    """
    cf_results   = state.get("counterfactual_results") or []
    interventions = state.get("interventions") or []
    memory       = state.get("memory_match")
    critique     = state.get("critique", {})
    retry_count  = state.get("retry_count", 0)

    # Pick highest-confidence confirmed root cause
    root = next(
        (r for r in cf_results if r.get("is_root_cause")), None
    )

    if root:
        confidence   = root["confidence"]
        root_cause   = root["candidate"]
        intervention = interventions[0]["action"] if interventions else "See ranked interventions"
        ttr_minutes  = interventions[0].get("ttr_minutes", 0) if interventions else 0
    elif memory:
        confidence   = memory["confidence"]
        root_cause   = memory["root_cause"]
        intervention = memory["resolution"]
        ttr_minutes  = memory["ttr_minutes"]
    else:
        confidence   = 0.0
        root_cause   = "Could not determine root cause"
        intervention = "Manual investigation required"
        ttr_minutes  = 0

    escalate = (
        confidence < settings.confidence_threshold
        or critique.get("verdict") == "FAIL"
        or retry_count >= settings.max_retries
    )

    verdict = {
        "root_cause":    root_cause,
        "confidence":    round(confidence, 2),
        "intervention":  intervention,
        "ttr_minutes":   ttr_minutes,
        "escalate":      escalate,
        "retries_used":  retry_count,
        "memory_used":   bool(memory and state.get("fast_path")),
    }

    log.info(
        "supervisor_finalize",
        confidence=confidence,
        escalate=escalate,
        root_cause=root_cause[:60] if root_cause else "",
    )

    flag = "🚨 Escalated to human" if escalate else "✅ Verdict delivered"
    return {
        "verdict":  verdict,
        "escalate": escalate,
        "steps":    [f"{flag}: confidence={confidence:.2f}, retries={retry_count}"],
    }
