"""
Intervention Ranker Agent — Structured output agent

Agentic practices demonstrated:
  - Structured output: enforces JSON schema, no free-form text
  - Tool grounding: every rank backed by causal confidence score
  - Format enforcement: output is always parseable by downstream consumers
"""

import json
import re
from typing import Any, Dict, List

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from src.utils.config import settings
from src.utils.logging import get_logger
from src.utils.state import RCAState

log = get_logger("intervention_ranker")

# Known TTR estimates per intervention type (minutes)
TTR_ESTIMATES = {
    "revert":   4,
    "restart":  3,
    "increase": 12,
    "kill":     2,
    "scale":    8,
    "patch":    20,
    "default":  10,
}

RISK_MAP = {
    "revert":   "low",
    "restart":  "low",
    "kill":     "low",
    "increase": "medium",
    "scale":    "medium",
    "patch":    "high",
    "default":  "medium",
}


def _estimate_ttr(action: str) -> int:
    for key in TTR_ESTIMATES:
        if key in action.lower():
            return TTR_ESTIMATES[key]
    return TTR_ESTIMATES["default"]


def _estimate_risk(action: str) -> str:
    for key in RISK_MAP:
        if key in action.lower():
            return RISK_MAP[key]
    return RISK_MAP["default"]


def _build_llm() -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        num_predict=512,
        temperature=0.0,
        format="json",
        timeout=settings.llm_timeout,
    )


def _fallback_interventions(cf_results: List[Dict], memory: Dict) -> List[Dict]:
    """Returns a guaranteed-valid intervention list if LLM output fails."""
    interventions = []

    if memory and memory.get("resolution"):
        res = memory["resolution"]
        interventions.append({
            "rank":        1,
            "action":      res,
            "confidence":  memory.get("confidence", 0.85),
            "ttr_minutes": memory.get("ttr_minutes", 5),
            "risk":        _estimate_risk(res),
            "note":        "From memory match",
        })

    for i, r in enumerate(cf_results):
        if r.get("is_root_cause"):
            action = f"Investigate and remediate: {r['candidate'][:80]}"
            interventions.append({
                "rank":        len(interventions) + 1,
                "action":      action,
                "confidence":  r["confidence"],
                "ttr_minutes": _estimate_ttr(action),
                "risk":        "medium",
                "note":        f"Causal confidence {r['confidence']}",
            })

    if not interventions:
        interventions.append({
            "rank": 1, "action": "Manual investigation required",
            "confidence": 0.0, "ttr_minutes": 30, "risk": "unknown", "note": "",
        })

    return interventions


def node_rank_interventions(state: RCAState) -> Dict[str, Any]:
    """
    Generates a ranked JSON list of interventions.
    Uses Ollama JSON mode to guarantee parseable output.
    Falls back to rule-based ranking if LLM output is malformed.
    """
    cf_results   = state.get("counterfactual_results") or []
    memory       = state.get("memory_match")
    evidence     = state.get("evidence", {})
    critique     = state.get("critique", {})

    confirmed = [r for r in cf_results if r.get("is_root_cause")]
    best      = max(confirmed, key=lambda r: r["confidence"]) if confirmed else None

    log.info("ranker_start", confirmed_causes=len(confirmed))

    llm = _build_llm()

    context = {
        "root_cause":     best["candidate"] if best else (memory["root_cause"] if memory else "unknown"),
        "confidence":     best["confidence"] if best else (memory["confidence"] if memory else 0),
        "downstream":     best.get("downstream_effects", [])[:4] if best else [],
        "services":       evidence.get("services", []),
        "deploys":        evidence.get("deploys", []),
        "memory_fix":     memory["resolution"] if memory else None,
        "critic_verdict": critique.get("verdict", "PASS"),
    }

    prompt = (
        "You are an SRE generating a ranked action plan for an incident. "
        "Respond ONLY with valid JSON — no markdown, no preamble.\n\n"
        f"Root cause: {context['root_cause']}\n"
        f"Confidence: {context['confidence']}\n"
        f"Downstream effects: {context['downstream']}\n"
        f"Affected services: {context['services']}\n"
        f"Recent deploys: {context['deploys']}\n"
        f"Memory-based fix suggestion: {context['memory_fix']}\n\n"
        "Generate a ranked list of 2-3 concrete interventions. "
        "Each must have: rank (int), action (str), confidence (float 0-1), "
        "ttr_minutes (int), risk (low/medium/high), note (str).\n\n"
        '{"interventions": [{"rank": 1, "action": "...", "confidence": 0.0, '
        '"ttr_minutes": 0, "risk": "low", "note": "..."}]}'
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw      = response.content.strip()
        clean    = re.sub(r"```json|```", "", raw).strip()
        parsed   = json.loads(clean)
        items    = parsed.get("interventions", [])

        # Enrich with computed fields
        for item in items:
            if "ttr_minutes" not in item or item["ttr_minutes"] == 0:
                item["ttr_minutes"] = _estimate_ttr(item.get("action", ""))
            if "risk" not in item:
                item["risk"] = _estimate_risk(item.get("action", ""))

        interventions = sorted(items, key=lambda x: x.get("rank", 99))
        if not interventions:
            raise ValueError("LLM returned empty interventions list")

    except Exception as e:
        log.warning("ranker_llm_fallback", error=str(e))
        interventions = _fallback_interventions(cf_results, memory)

    log.info("ranker_complete", count=len(interventions))

    summary = f"Rank 1: {interventions[0]['action'][:60]}" if interventions else "No interventions"
    return {
        "interventions": interventions,
        "steps": [f"📋 Intervention Ranker: {len(interventions)} actions ranked. {summary}"],
    }
