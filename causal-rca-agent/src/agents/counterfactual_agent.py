"""
Counterfactual Agent — ReAct loop agent

Agentic practices demonstrated:
  - ReAct loop: Reason → Act (call do() tool) → Observe → Reason again
  - Self-termination: stops when confidence > threshold OR candidates exhausted
  - Tool use: do() operator called per candidate
"""

from typing import Any, Dict, List

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from src.tools.causal_tools import run_counterfactual
from src.utils.config import settings
from src.utils.logging import get_logger
from src.utils.state import RCAState

log = get_logger("counterfactual_agent")


def _build_llm() -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model_large,
        num_predict=128,
        temperature=0.0,
        timeout=settings.llm_timeout,
    )


def node_counterfactual(state: RCAState) -> Dict[str, Any]:
    """
    ReAct loop over root cause candidates.

    For each candidate:
      Reason: should I test this? (LLM ranks candidates)
      Act:    call run_counterfactual(candidate)
      Observe: read confidence score
      Reason: is confidence > threshold? → stop; else → next candidate
    """
    dag       = state["causal_dag"]
    evidence  = state["evidence"]
    memory    = state.get("memory_match")
    candidates = dag.get("root_candidates", [])

    # If memory suggests a candidate, prioritise it
    if memory and memory.get("root_cause"):
        mem_hint = memory["root_cause"].split()[0].lower()
        candidates = sorted(
            candidates,
            key=lambda c: 0 if mem_hint in c.lower() else 1,
        )

    log.info("counterfactual_start", candidates=candidates)

    llm     = _build_llm()
    results: List[Dict] = []
    step_log: List[str] = []

    for i, candidate in enumerate(candidates[:settings.max_retries + 1]):
        # ReAct: Reason — should we test this candidate?
        reason_prompt = (
            f"Incident context: {evidence.get('services', [])}. "
            f"Deploys in window: {evidence.get('deploys', [])}. "
            f"We are about to run a counterfactual test on: '{candidate}'. "
            f"In one sentence, why is this a plausible root cause candidate?"
        )
        reason_resp = llm.invoke([HumanMessage(content=reason_prompt)])
        reasoning   = reason_resp.content.strip()

        # ReAct: Act — call do() operator tool
        cf_result = run_counterfactual(candidate, dag, evidence)
        cf_result["reasoning"] = reasoning
        results.append(cf_result)

        log.info(
            "counterfactual_result",
            candidate=candidate,
            confidence=cf_result["confidence"],
            verdict=cf_result["verdict"],
        )

        step_log.append(
            f"🔎 Counterfactual [{i+1}]: '{candidate[:50]}' → "
            f"{cf_result['verdict']} (confidence: {cf_result['confidence']}). "
            f"{reasoning}"
        )

        # ReAct: Observe — stop if root cause confirmed above threshold
        if cf_result["is_root_cause"] and cf_result["confidence"] >= settings.confidence_threshold:
            log.info("counterfactual_stop", reason="confidence_threshold_met", iterations=i + 1)
            break

    return {
        "counterfactual_results": results,
        "steps": step_log,
    }
