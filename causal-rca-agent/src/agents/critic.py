"""
Critic Agent — Self-reflection / quality gate

Agentic practices demonstrated:
  - Self-reflection: independently reviews verdict before it ships
  - Hallucination prevention: checks every claim is grounded in evidence
  - Quality gate: can reject and send pipeline back for another pass
"""

from typing import Any, Dict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from src.utils.config import settings
from src.utils.logging import get_logger
from src.utils.state import RCAState

log = get_logger("critic")


def _build_llm() -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        num_predict=256,
        temperature=0.0,
        format="json",
        timeout=settings.llm_timeout,
    )


def node_critic(state: RCAState) -> Dict[str, Any]:
    """
    Reviews counterfactual results and causal DAG.
    Returns PASS or FAIL with a reason.
    FAIL sends the pipeline back for another retry via increment_retry node.
    """
    cf_results = state.get("counterfactual_results") or []
    dag        = state.get("causal_dag", {})
    evidence   = state.get("evidence", {})
    memory     = state.get("memory_match")

    # Fast-path: if memory match is very strong, trust it
    if memory and memory["similarity"] >= 0.90 and state.get("fast_path"):
        log.info("critic_fast_path", similarity=memory["similarity"])
        return {
            "critique": {
                "verdict": "PASS",
                "score":   memory["confidence"],
                "reason":  f"Memory match similarity {memory['similarity']} exceeds fast-path threshold. Past resolution verified.",
            },
            "steps": [f"🛡️ Critic: PASS (fast-path memory match, similarity={memory['similarity']})"],
        }

    confirmed = [r for r in cf_results if r.get("is_root_cause")]
    if not confirmed:
        return {
            "critique": {
                "verdict": "FAIL",
                "score":   0.0,
                "reason":  "No root cause confirmed by counterfactual testing.",
            },
            "steps": ["🛡️ Critic: FAIL — no confirmed root cause found"],
        }

    best       = max(confirmed, key=lambda r: r["confidence"])
    confidence = best["confidence"]
    candidate  = best["candidate"]
    downstream = best.get("downstream_effects", [])
    edges      = dag.get("edges", [])
    services   = evidence.get("services", [])

    llm = _build_llm()
    prompt = (
        "You are a senior SRE reviewing an incident root cause analysis. "
        "Respond ONLY with valid JSON — no extra text.\n\n"
        f"Proposed root cause: {candidate}\n"
        f"Confidence score: {confidence}\n"
        f"Downstream effects attributed to this cause: {downstream[:4]}\n"
        f"Causal edges in evidence: {len(edges)}\n"
        f"Affected services: {services}\n\n"
        "Review this verdict. Does the evidence support the root cause claim? "
        "Are there obvious alternative causes not tested?\n\n"
        'Respond with exactly: {"verdict": "PASS" or "FAIL", '
        '"score": 0.0-1.0, "reason": "one sentence"}'
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    raw      = response.content.strip()

    try:
        import json, re
        clean   = re.sub(r"```json|```", "", raw).strip()
        parsed  = json.loads(clean)
        verdict = parsed.get("verdict", "FAIL").upper()
        score   = float(parsed.get("score", 0.5))
        reason  = parsed.get("reason", "No reason provided")
    except Exception:
        # If LLM output can't be parsed, default based on confidence
        verdict = "PASS" if confidence >= settings.confidence_threshold else "FAIL"
        score   = confidence
        reason  = "JSON parse failed — defaulting to confidence-based verdict"

    log.info("critic_verdict", verdict=verdict, score=score)

    icon = "✅" if verdict == "PASS" else "❌"
    return {
        "critique": {"verdict": verdict, "score": score, "reason": reason},
        "steps":    [f"🛡️ Critic: {icon} {verdict} (score={score:.2f}) — {reason}"],
    }
